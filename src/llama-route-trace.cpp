#include "llama-route-trace.h"

#include "ggml.h"
#include "ggml-backend.h"

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

bool parse_route_layer(const ggml_tensor * tensor, uint16_t & layer) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_I32) {
        return false;
    }

    const size_t prefix_len = std::strlen(ROUTE_TENSOR_PREFIX);
    if (std::strncmp(tensor->name, ROUTE_TENSOR_PREFIX, prefix_len) != 0) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(tensor->name + prefix_len, &end, 10);
    if (errno != 0 || end == tensor->name + prefix_len || *end != '\0' ||
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

        std::vector<uint8_t> record;
        record.reserve(4 + static_cast<size_t>(n_rows) * top_k * sizeof(uint16_t));
        append_u16_le(record, layer);
        append_u16_le(record, n_rows);

        for (uint16_t row = 0; row < n_rows; ++row) {
            for (uint16_t rank = 0; rank < top_k; ++rank) {
                int32_t expert = -1;
                std::memcpy(&expert, source + static_cast<size_t>(row) * tensor->nb[1] +
                        static_cast<size_t>(rank) * tensor->nb[0], sizeof(expert));
                if (expert < 0 || expert >= n_experts) {
                    report_error("route tensor contains an out-of-range expert ID");
                    return;
                }
                append_u16_le(record, static_cast<uint16_t>(expert));
            }
        }

        if (std::fwrite(record.data(), 1, record.size(), file_) != record.size()) {
            report_error("failed to write route record");
        }
        // Serving processes are routinely force-killed (never reach atexit),
        // so flush periodically to keep collected traces recoverable.
        if (++records_since_flush_ >= 512) {
            std::fflush(file_);
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
                report_error("route parameters changed after the trace was opened");
                return false;
            }
            return true;
        }

        file_ = std::fopen(route_trace_path(), "wb");
        if (file_ == nullptr) {
            report_error("could not open LONGSPEAR_ROUTE_TRACE path");
            return false;
        }

        std::setvbuf(file_, nullptr, _IOFBF, 1024 * 1024);
        n_experts_     = n_experts;
        top_k_         = top_k;
        n_layers_      = n_layers;
        n_main_layers_ = n_main_layers;

        if (std::fprintf(file_,
                "LONGSPEAR_ROUTE_TRACE v1 model=qwen4exp experts=%u top_k=%u layers=%u main_layers=%u "
                "endian=little record=u16_layer,u16_n_rows,n_rows*top_k*u16_expert\n",
                n_experts, top_k, n_layers, n_main_layers) < 0) {
            report_error("failed to write route trace header");
            return false;
        }
        return true;
    }

    void report_error(const char * message) {
        if (!reported_) {
            std::fprintf(stderr, "longspear route trace: %s: %s\n", message,
                    errno != 0 ? std::strerror(errno) : "trace disabled");
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

} // namespace

bool llama_route_trace_enabled() {
    return route_trace_path() != nullptr;
}

void llama_route_trace_mark_output(ggml_tensor * tensor, const char * name, uint16_t top_k) {
    if (!llama_route_trace_enabled() || tensor == nullptr || name == nullptr ||
            std::strcmp(name, "ffn_moe_topk") != 0 || tensor->type != GGML_TYPE_I32 ||
            tensor->ne[0] != top_k) {
        return;
    }
    ggml_set_output(tensor);
}

void llama_route_trace_collect(
        ggml_backend_sched * sched,
        ggml_cgraph *        graph,
        uint16_t             n_experts,
        uint16_t             top_k,
        uint16_t             n_layers,
        uint16_t             n_main_layers) {
    if (!llama_route_trace_enabled() || sched == nullptr || graph == nullptr) {
        return;
    }

    std::vector<pending_route> pending;
    pending.reserve(n_layers);

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * tensor = graph->nodes[i];
        uint16_t layer = 0;
        if (!parse_route_layer(tensor, layer) || tensor->ne[0] != top_k ||
                tensor->ne[1] <= 0 || tensor->ne[1] > std::numeric_limits<uint16_t>::max() ||
                tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        if (backend == nullptr) {
            continue;
        }

        pending.push_back({ tensor, layer, static_cast<uint16_t>(tensor->ne[1]),
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
        writer().write(item.layer, item.n_rows, n_experts, top_k, n_layers, n_main_layers,
                item.data.data(), item.tensor);
    }
}
