#pragma once

#include <cstdint>

struct ggml_backend_sched;
struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

struct llama_route_trace_pass {
    uint16_t n_rows = 0;
    uint16_t n_experts = 0;
    uint16_t top_k = 0;
    uint16_t n_layers = 0;
    uint16_t n_main_layers = 0;
    bool enabled = false;
    bool full = false;
};

bool llama_route_trace_enabled();

// Snapshot routed IDs into a dedicated terminal graph output immediately after
// top-k selection. No-op unless LONGSPEAR_ROUTE_TRACE_FULL=1.
void llama_route_trace_capture(
        struct ggml_context * ctx,
        struct ggml_cgraph *  graph,
        struct ggml_tensor *  tensor,
        int                   layer,
        uint16_t              top_k);

// Open and flush the header before graph submission. Default mode registers a
// callback for IDs already copied host-side by selective CPU-MoE offload.
void llama_route_trace_begin(
        struct ggml_backend_sched * sched,
        llama_route_trace_pass &    pass,
        uint16_t                    n_rows,
        uint16_t                    n_experts,
        uint16_t                    top_k,
        uint16_t                    n_layers,
        uint16_t                    n_main_layers);

// In LONGSPEAR_ROUTE_TRACE_FULL=1 mode, collect all marked tensors after graph
// submission and pay one scheduler-wide synchronization.
void llama_route_trace_collect(
        struct ggml_backend_sched * sched,
        struct ggml_cgraph *        graph,
        const llama_route_trace_pass & pass);

void llama_route_trace_end(
        struct ggml_backend_sched * sched,
        const llama_route_trace_pass & pass);
