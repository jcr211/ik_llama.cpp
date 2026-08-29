#include "llama-route-trace.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace {

constexpr const char * ROUTE_TENSOR_PREFIX = "ffn_moe_topk-";

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

bool parse_route_layer(const ggml_tensor * tensor, uint16_t & layer) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_I32) {
        return false;
    }

    const char * prefix = std::strstr(tensor->name, ROUTE_TENSOR_PREFIX);
    if (prefix == nullptr) {
        return false;
    }
    prefix += std::strlen(ROUTE_TENSOR_PREFIX);

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

        std::vector<uint8_t> payload;
        payload.reserve(static_cast<size_t>(n_rows) * top_k * sizeof(uint16_t));
        uint16_t rows_written = 0;

        for (uint16_t row = 0; row < n_rows; ++row) {
            const size_t row_begin = payload.size();
            bool valid = true;
            for (uint16_t rank = 0; rank < top_k; ++rank) {
                int32_t expert = -1;
                std::memcpy(&expert, source + static_cast<size_t>(row) * tensor->nb[1] +
                        static_cast<size_t>(rank) * tensor->nb[0], sizeof(expert));
                if (expert < 0 || expert >= n_experts) {
                    payload.resize(row_begin);
                    note_bad_row(layer, expert);
                    valid = false;
                    break;
                }
                append_u16_le(payload, static_cast<uint16_t>(expert));
            }
            if (valid) {
                ++rows_written;
            }
        }

        if (rows_written == 0) {
            return;
        }

        std::vector<uint8_t> record;
        record.reserve(4 + payload.size());
        append_u16_le(record, layer);
        append_u16_le(record, rows_written);
        record.insert(record.end(), payload.begin(), payload.end());

        if (std::fwrite(record.data(), 1, record.size(), file_) != record.size()) {
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
            !parse_route_layer(tensor, layer) || layer >= pass.n_main_layers ||
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

void llama_route_trace_mark_output(ggml_tensor * tensor, const char * name, uint16_t top_k) {
    if (!llama_route_trace_enabled() || !route_trace_full() || tensor == nullptr || name == nullptr ||
            std::strcmp(name, "ffn_moe_topk") != 0 || tensor->type != GGML_TYPE_I32 ||
            tensor->ne[0] != top_k) {
        return;
    }
    ggml_set_output(tensor);
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
        if (!parse_route_layer(tensor, layer) || tensor->ne[0] != pass.top_k ||
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
        pending_route & item = pending.back();
        ggml_backend_tensor_get_async(backend, tensor, item.data.data(), 0, item.data.size());
    }

    if (pending.empty()) {
        return;
    }

    // One gated synchronization covers every GPU-resident layer readback in the pass.
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
