#pragma once

#include "ggml.h"

// Layout of the qwen4exp PLE convolution history, shared by the graph, the per-step speculative
// checkpoint and its test. A PLE layer's state row ends with hist*hc_dim floats laid out
// [hist, hc_dim]. The graph prepends them to the ubatch: conv_in = concat(history, xt) is
// [hist + n_tokens, hc_dim], so the history after the token at conv_in column c is the window of
// hist columns ending at c.

// columns [start, start + hist) of conv_in
static inline ggml_tensor * llama_ple_conv_window(ggml_context * ctx, ggml_tensor * conv_in,
        int64_t start, int32_t hist) {
    return ggml_view_2d(ctx, conv_in, hist, conv_in->ne[1], conv_in->nb[1], start*conv_in->nb[0]);
}

// byte offset of the history in a state row of ne0 elements
static inline size_t llama_ple_conv_row_offset(int64_t ne0, int32_t hist, int32_t hc_dim, size_t esz) {
    return esz*(size_t) (ne0 - (int64_t) hist*hc_dim);
}

// slot j of a per-step buffer holds the history after verify token j: the same index convention as
// the delta-net per-step states, where slot j is the state after token j
static inline ggml_tensor * llama_ple_conv_per_step_slot(ggml_context * ctx, ggml_tensor * per_step,
        int32_t hist, int32_t hc_dim, int32_t j) {
    const size_t esz = ggml_element_size(per_step);
    return ggml_view_2d(ctx, per_step, hist, hc_dim, hist*esz, (size_t) j*hist*hc_dim*esz);
}

// after token j the history is conv_in columns [j + 1, j + 1 + hist); the last token needs no slot
// because a fully accepted batch keeps the state row the graph writes anyway
static inline void llama_ple_conv_save_per_step(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * conv_in,
        ggml_tensor * per_step, int32_t hist, int32_t hc_dim, int32_t n_tokens) {
    for (int32_t j = 0; j + 1 < n_tokens; ++j) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx,
                llama_ple_conv_window(ctx, conv_in, j + 1, hist),
                llama_ple_conv_per_step_slot(ctx, per_step, hist, hc_dim, j)));
    }
}
