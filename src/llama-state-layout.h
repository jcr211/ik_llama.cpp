#pragma once

// State-OS layout descriptor (llama_state_seq_layout_desc), rendered from plain values so it can be golden-tested
// without a model (tests/test-stateos-layout.cpp).
//
// BUMP RULE: the kv line is hashed into the State-OS "kv_geometry" hard field. Any change to what this renders, or to
// what write_kv_cache/write_kv_cache_data/read_kv_cache_data put in or expect from a sequence-state payload, must
// (1) update the golden strings in tests/test-stateos-layout.cpp and (2) bump STATEOS_KV_LAYOUT_VERSION in
// examples/server/stateos-header.h when saved states from the old layout must no longer load. A payload whose bytes
// keep their layout but change meaning (e.g. a rotation implementation) is only caught by that version bump.

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct llama_state_layout_row {
    uint32_t il;
    int      type;
    uint64_t width; // row bytes (K), embedding width (V), elements (S, indexer)
};

struct llama_state_layout_info {
    // rope line
    int      rope_type        = 0;
    float    rope_freq_base   = 0.0f;
    float    rope_freq_scale  = 0.0f;
    uint32_t n_ctx_orig_yarn  = 0;
    float    yarn_ext_factor  = 0.0f;
    float    yarn_attn_factor = 0.0f;
    float    yarn_beta_fast   = 0.0f;
    float    yarn_beta_slow   = 0.0f;

    // kv line
    std::string arch;
    int      seq_version  = 0;
    uint32_t n_ctx        = 0;
    uint32_t kv_size      = 0;
    uint32_t v_state      = 0; // 0 plain V, 1 transposed V, 2 no V
    uint32_t n_layer      = 0;
    int      flash_attn   = 0;
    int      mla_attn     = 0;
    int      k_hadamard   = 0;
    int      v_hadamard   = 0;
    int      idx_hadamard = 0;
    bool     compacted    = false;
    uint32_t size_swa     = 0;
    uint32_t sink_rows    = 0;
    std::vector<llama_state_layout_row> k_rows; // layers with a K cache
    std::vector<llama_state_layout_row> v_rows; // layers with a V cache (ignored when v_state == 2)
    int      qnext        = 0;
    std::vector<llama_state_layout_row> s_rows; // recurrent state tensors
    int      has_indexer  = 0;
    std::vector<llama_state_layout_row> r_rows; // indexer key caches
};

static inline std::string llama_state_layout_fmt(const char * fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return std::string(buf, n < 0 ? 0 : (size_t) std::min(n, (int) sizeof(buf) - 1));
}

// "rope=...\nkv=...\n" — the exact text llama_state_seq_layout_desc returns
static inline std::string llama_state_layout_render(const llama_state_layout_info & L) {
    std::string out = llama_state_layout_fmt(
            "rope=type=%d base=%.9g scale=%.9g orig_yarn=%u ext=%.9g attn=%.9g beta_fast=%.9g beta_slow=%.9g\n",
            L.rope_type, L.rope_freq_base, L.rope_freq_scale, L.n_ctx_orig_yarn,
            L.yarn_ext_factor, L.yarn_attn_factor, L.yarn_beta_fast, L.yarn_beta_slow);

    out += llama_state_layout_fmt(
            "kv=arch=%s seqv=%d n_ctx=%u size=%u v_state=%u n_layer=%u fa=%d mla=%d khad=%d vhad=%d ihad=%d compact=%d",
            L.arch.c_str(), L.seq_version, L.n_ctx, L.kv_size, L.v_state, L.n_layer,
            L.flash_attn, L.mla_attn, L.k_hadamard, L.v_hadamard, L.idx_hadamard, (int) L.compacted);
    if (L.compacted) {
        out += llama_state_layout_fmt(" size_swa=%u sink_rows=%u", L.size_swa, L.sink_rows);
    }
    // K and V rows interleave per layer, in layer order
    size_t iv = 0;
    for (const auto & k : L.k_rows) {
        out += llama_state_layout_fmt(" k%u=%d/%llu", k.il, k.type, (unsigned long long) k.width);
        while (iv < L.v_rows.size() && L.v_rows[iv].il < k.il) {
            ++iv;
        }
        if (L.v_state != 2 && iv < L.v_rows.size() && L.v_rows[iv].il == k.il) {
            out += llama_state_layout_fmt(" v%u=%d/%u", L.v_rows[iv].il, L.v_rows[iv].type, (unsigned) L.v_rows[iv].width);
        }
    }
    out += llama_state_layout_fmt(" qnext=%d", L.qnext);
    for (const auto & s : L.s_rows) {
        out += llama_state_layout_fmt(" s%u=%d/%lld", s.il, s.type, (long long) s.width);
    }
    out += llama_state_layout_fmt(" idx=%d", L.has_indexer);
    for (const auto & r : L.r_rows) {
        out += llama_state_layout_fmt(" r%u=%d/%lld", r.il, r.type, (long long) r.width);
    }
    out += "\n";
    return out;
}
