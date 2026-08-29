#include "llama-route-trace.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace {

constexpr const char * ROUTE_TENSOR_PREFIX   = "ffn_moe_topk-";
constexpr const char * CAPTURE_TENSOR_PREFIX = "route_trace_topk-";

const char * route_trace_path() {
    static const char * path = [] {
        const char * value = std::getenv("LONGSPEAR_ROUTE_TRACE");
        return value != nullptr && value[0] != '\0' ? value : nullptr;
    }();
    return path;
}

bool route_trace_full() {
    static const bool full = [] {
        const char * value = std::getenv("LONGSPEAR_ROUTE_TRACE_FULL");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return full;
}

bool parse_route_layer(const ggml_tensor * tensor, const char * tensor_prefix, uint16_t & layer) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_I32) {
        return false;
    }

    const char * prefix = std::strstr(tensor->name, tensor_prefix);
    if (prefix == nullptr) {
        return false;
    }
    prefix += std::strlen(tensor_prefix);

    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(prefix, &end, 10);
    if (errno != 0 || end == prefix || (*end != '\0' && *end != '#') ||
            parsed < 0 || parsed > std::numeric_limits<uint16_t>::max()) {
        return false;
    }

    layer = static_cast<uint16_t>(parsed);
    return true;
}

void append_u16_le(std::vector<uint8_t> & dst, uint16_t value) {
    dst.push_back(static_cast<uint8_t>(value & 0xff));
    dst.push_back(static_cast<uint8_t>(value >> 8));
}

class route_trace_writer {
public:
    ~route_trace_writer() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_ != nullptr) {
            std::fflush(file_);
            std::fclose(file_);
            file_ = nullptr;
        }
        if (route_trace_full() && !layer_rows_.empty()) {
            report_full_self_check();
        }
        if (rows_skipped_ != 0) {
            std::fprintf(stderr,
                    "longspear route trace: rows_skipped=%llu first_bad_layer=%u first_bad_value=%d\n",
                    static_cast<unsigned long long>(rows_skipped_), first_bad_layer_, first_bad_value_);
        }
    }

    bool begin(uint16_t n_experts, uint16_t top_k, uint16_t n_layers, uint16_t n_main_layers) {
        std::lock_guard<std::mutex> lock(mutex_);
        return open(n_experts, top_k, n_layers, n_main_layers);
    }

    void disable(const char * message) {
        std::lock_guard<std::mutex> lock(mutex_);
        report_nonfatal_error(message);
        failed_ = true;
    }

    void write(
            uint16_t            layer,
            uint16_t            n_rows,
            uint16_t            n_experts,
            uint16_t            top_k,
            uint16_t            n_layers,
            uint16_t            n_main_layers,
            const uint8_t *     source,
            const ggml_tensor * tensor) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!open(n_experts, top_k, n_layers, n_main_layers)) {
            return;
        }
        if (layer >= n_layers_) {
            report_nonfatal_error("route record layer is outside the declared layer count");
            return;
        }

        payload_.clear();
        payload_.reserve(static_cast<size_t>(n_rows) * top_k * sizeof(uint16_t));
        row_experts_.resize(top_k);
        uint16_t rows_written = 0;

        for (uint16_t row = 0; row < n_rows; ++row) {
            bool valid = true;
            for (uint16_t rank = 0; rank < top_k; ++rank) {
                int32_t expert = -1;
                std::memcpy(&expert, source + static_cast<size_t>(row) * tensor->nb[1] +
                        static_cast<size_t>(rank) * tensor->nb[0], sizeof(expert));
                if (expert < 0 || expert >= n_experts) {
                    note_bad_row(layer, expert);
                    valid = false;
                    break;
                }
                row_experts_[rank] = static_cast<uint16_t>(expert);
            }
            if (valid) {
                for (uint16_t expert : row_experts_) {
                    append_u16_le(payload_, expert);
                    if (!expert_frequencies_.empty()) {
                        ++expert_frequencies_[static_cast<size_t>(layer) * n_experts_ + expert];
                    }
                }
                if (!layer_rows_.empty()) {
                    ++layer_rows_[layer];
                }
                ++rows_written;
            }
        }

        if (rows_written == 0) {
            return;
        }

        record_.clear();
        record_.reserve(4 + payload_.size());
        append_u16_le(record_, layer);
        append_u16_le(record_, rows_written);
        record_.insert(record_.end(), payload_.begin(), payload_.end());

        if (std::fwrite(record_.data(), 1, record_.size(), file_) != record_.size()) {
            report_io_error("failed to write route record");
            return;
        }
        // Serving processes are routinely force-killed (never reach atexit),
        // so flush periodically to keep collected traces recoverable.
        if (++records_since_flush_ >= 512) {
            if (std::fflush(file_) != 0) {
                report_io_error("failed to flush route trace records");
                return;
            }
            records_since_flush_ = 0;
        }
    }

private:
    bool open(uint16_t n_experts, uint16_t top_k, uint16_t n_layers, uint16_t n_main_layers) {
        if (failed_) {
            return false;
        }
        if (file_ != nullptr) {
            if (n_experts != n_experts_ || top_k != top_k_ ||
                    n_layers != n_layers_ || n_main_layers != n_main_layers_) {
                report_nonfatal_error("route parameters changed after the trace was opened");
                return false;
            }
            return true;
        }

        file_ = std::fopen(route_trace_path(), "wb");
        if (file_ == nullptr) {
            report_io_error("could not open LONGSPEAR_ROUTE_TRACE path");
            return false;
        }

        std::setvbuf(file_, nullptr, _IOFBF, 1024 * 1024);
        n_experts_     = n_experts;
        top_k_         = top_k;
        n_layers_      = n_layers;
        n_main_layers_ = n_main_layers;
        if (route_trace_full()) {
            expert_frequencies_.assign(static_cast<size_t>(n_layers) * n_experts, 0);
            layer_rows_.assign(n_layers, 0);
        }

        if (std::fprintf(file_,
                "LONGSPEAR_ROUTE_TRACE v1 model=qwen4exp experts=%u top_k=%u layers=%u main_layers=%u "
                "mode=%s endian=little record=u16_layer,u16_n_rows,n_rows*top_k*u16_expert\n",
                n_experts, top_k, n_layers, n_main_layers, route_trace_full() ? "full" : "offload-host") < 0) {
            report_io_error("failed to write route trace header");
            return false;
        }
        if (std::fflush(file_) != 0) {
            report_io_error("failed to flush route trace header");
            return false;
        }
        return true;
    }

    void note_bad_row(uint16_t layer, int32_t value) {
        if (rows_skipped_++ == 0) {
            first_bad_layer_ = layer;
            first_bad_value_ = value;
        }
    }

    void report_full_self_check() const {
        uint64_t identical_pairs = 0;
        uint64_t compared_pairs  = 0;
        uint32_t observed_layers = 0;

        for (uint16_t lhs = 0; lhs < n_layers_; ++lhs) {
            if (layer_rows_[lhs] == 0) {
                continue;
            }
            ++observed_layers;
            const auto lhs_begin = expert_frequencies_.begin() + static_cast<size_t>(lhs) * n_experts_;
            for (uint16_t rhs = lhs + 1; rhs < n_layers_; ++rhs) {
                if (layer_rows_[rhs] == 0) {
                    continue;
                }
                ++compared_pairs;
                const auto rhs_begin = expert_frequencies_.begin() + static_cast<size_t>(rhs) * n_experts_;
                if (std::equal(lhs_begin, lhs_begin + n_experts_, rhs_begin)) {
                    ++identical_pairs;
                }
            }
        }

        std::fprintf(stderr,
                "longspear route trace: full_self_check identical_layer_pairs=%llu "
                "compared_layer_pairs=%llu observed_layers=%u\n",
                static_cast<unsigned long long>(identical_pairs),
                static_cast<unsigned long long>(compared_pairs), observed_layers);
    }

    void report_nonfatal_error(const char * message) {
        if (!reported_) {
            std::fprintf(stderr, "longspear route trace: %s\n", message);
            reported_ = true;
        }
    }

    void report_io_error(const char * message) {
        if (!reported_) {
            std::fprintf(stderr, "longspear route trace: %s: %s\n", message,
                    errno != 0 ? std::strerror(errno) : "I/O failure; trace disabled");
            reported_ = true;
        }
        failed_ = true;
    }

    std::mutex mutex_;
    FILE * file_ = nullptr;
    uint16_t n_experts_ = 0;
    uint16_t top_k_ = 0;
    uint16_t n_layers_ = 0;
    uint16_t n_main_layers_ = 0;
    bool failed_ = false;
    bool reported_ = false;
    uint32_t records_since_flush_ = 0;
    uint64_t rows_skipped_ = 0;
    uint16_t first_bad_layer_ = 0;
    int32_t first_bad_value_ = 0;
    std::vector<uint8_t> payload_;
    std::vector<uint8_t> record_;
    std::vector<uint16_t> row_experts_;
    std::vector<uint64_t> expert_frequencies_;
    std::vector<uint64_t> layer_rows_;
};

struct pending_route {
    ggml_tensor * tensor;
    uint16_t layer;
    uint16_t n_rows;
    std::vector<uint8_t> data;
};

route_trace_writer & writer() {
    static route_trace_writer instance;
    return instance;
}

void collect_host_moe_ids(const ggml_tensor * tensor, const int32_t * data, void * user_data) {
    const llama_route_trace_pass & pass = *static_cast<const llama_route_trace_pass *>(user_data);
    uint16_t layer = 0;
    if (!pass.enabled || pass.full || tensor == nullptr || data == nullptr ||
            !parse_route_layer(tensor, ROUTE_TENSOR_PREFIX, layer) || layer >= pass.n_main_layers ||
            tensor->ne[0] != pass.top_k || tensor->ne[1] <= 0 ||
            tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        return;
    }

    const uint16_t n_rows = static_cast<uint16_t>(std::min<int64_t>(pass.n_rows, tensor->ne[1]));
    writer().write(layer, n_rows, pass.n_experts, pass.top_k, pass.n_layers, pass.n_main_layers,
            reinterpret_cast<const uint8_t *>(data), tensor);
}

} // namespace

bool llama_route_trace_enabled() {
    return route_trace_path() != nullptr;
}

void llama_route_trace_capture(
        ggml_context * ctx,
        ggml_cgraph *  graph,
        ggml_tensor *  tensor,
        int            layer,
        uint16_t       top_k) {
    if (!llama_route_trace_enabled() || !route_trace_full() || ctx == nullptr || graph == nullptr ||
            tensor == nullptr || layer < 0 || tensor->type != GGML_TYPE_I32 || tensor->ne[0] != top_k) {
        return;
    }

    ggml_tensor * capture = ggml_cpy(ctx, tensor, ggml_dup_tensor(ctx, tensor));
    ggml_format_name(capture, "%s%d", CAPTURE_TENSOR_PREFIX, layer);
    ggml_set_output(capture);
    ggml_build_forward_expand(graph, capture);
}

void llama_route_trace_begin(
        ggml_backend_sched *     sched,
        llama_route_trace_pass & pass,
        uint16_t                 n_rows,
        uint16_t                 n_experts,
        uint16_t                 top_k,
        uint16_t                 n_layers,
        uint16_t                 n_main_layers) {
    pass = { n_rows, n_experts, top_k, n_layers, n_main_layers, false, route_trace_full() };
    if (!llama_route_trace_enabled() || sched == nullptr || n_rows == 0 ||
            !writer().begin(n_experts, top_k, n_layers, n_main_layers)) {
        return;
    }

    pass.enabled = true;
    if (!pass.full) {
        ggml_backend_sched_set_moe_ids_callback(sched, collect_host_moe_ids, &pass);
    }
}

void llama_route_trace_collect(
        ggml_backend_sched * sched,
        ggml_cgraph *        graph,
        const llama_route_trace_pass & pass) {
    if (!pass.enabled || !pass.full || sched == nullptr || graph == nullptr) {
        return;
    }

    std::vector<pending_route> pending;
    pending.reserve(pass.n_layers);

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * tensor = graph->nodes[i];
        uint16_t layer = 0;
        if (!parse_route_layer(tensor, CAPTURE_TENSOR_PREFIX, layer) || layer >= pass.n_layers ||
                !(tensor->flags & GGML_TENSOR_FLAG_OUTPUT) || tensor->ne[0] != pass.top_k ||
                tensor->ne[1] <= 0 || tensor->ne[1] > std::numeric_limits<uint16_t>::max() ||
                tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        if (backend == nullptr) {
            continue;
        }

        pending.push_back({ tensor, layer,
                static_cast<uint16_t>(std::min<int64_t>(pass.n_rows, tensor->ne[1])),
                std::vector<uint8_t>(ggml_nbytes(tensor)) });
    }

    if (pending.empty()) {
        return;
    }

    // FULL captures must occupy disjoint retained storage. Refuse to emit a
    // trace rather than silently recording aliased arena contents.
    for (size_t i = 0; i < pending.size(); ++i) {
        const pending_route & lhs = pending[i];
        if (lhs.tensor->buffer == nullptr || lhs.tensor->data == nullptr) {
            writer().disable("full capture output has no allocated storage; trace disabled");
            return;
        }
        const uintptr_t lhs_begin = reinterpret_cast<uintptr_t>(lhs.tensor->data);
        const uintptr_t lhs_end   = lhs_begin + lhs.data.size();
        for (size_t j = i + 1; j < pending.size(); ++j) {
            const pending_route & rhs = pending[j];
            if (lhs.tensor->buffer != rhs.tensor->buffer || rhs.tensor->data == nullptr) {
                continue;
            }
            const uintptr_t rhs_begin = reinterpret_cast<uintptr_t>(rhs.tensor->data);
            const uintptr_t rhs_end   = rhs_begin + rhs.data.size();
            if (lhs_begin < rhs_end && rhs_begin < lhs_end) {
                writer().disable("full capture outputs overlap in arena storage; trace disabled");
                return;
            }
        }
    }

    // The CPU backend has no async-get implementation, so get_async falls back
    // to an immediate memcpy. Complete the graph first; this is also the only
    // safe observation boundary for a CUDA-graph replay.
    ggml_backend_sched_synchronize(sched);

    for (pending_route & item : pending) {
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, item.tensor);
        ggml_backend_tensor_get_async(backend, item.tensor, item.data.data(), 0, item.data.size());
    }

    // Drain device-to-host copies queued after the compute synchronization.
    ggml_backend_sched_synchronize(sched);

    for (const pending_route & item : pending) {
        writer().write(item.layer, item.n_rows, pass.n_experts, pass.top_k, pass.n_layers, pass.n_main_layers,
                item.data.data(), item.tensor);
    }
}

void llama_route_trace_end(
        ggml_backend_sched * sched,
        const llama_route_trace_pass & pass) {
    if (pass.enabled && !pass.full && sched != nullptr) {
        ggml_backend_sched_set_moe_ids_callback(sched, nullptr, nullptr);
    }
}
