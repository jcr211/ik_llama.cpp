#pragma once

#include <cstdint>

struct ggml_backend_sched;
struct ggml_cgraph;
struct ggml_tensor;

bool llama_route_trace_enabled();

// Keep routed IDs alive until the post-compute readback. This is a no-op when
// LONGSPEAR_ROUTE_TRACE is unset or the tensor is not the configured top-k.
void llama_route_trace_mark_output(
        struct ggml_tensor * tensor,
        const char *         name,
        uint16_t             top_k);

// Collect all marked qwen4exp route tensors from one decode/prefill graph pass.
void llama_route_trace_collect(
        struct ggml_backend_sched * sched,
        struct ggml_cgraph * graph,
        uint16_t             n_experts,
        uint16_t             top_k,
        uint16_t             n_layers,
        uint16_t             n_main_layers);
