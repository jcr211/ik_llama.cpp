// Golden test for the State-OS layout descriptor (src/llama-state-layout.h). Its kv line is hashed into the
// "kv_geometry" hard field of every saved state: if this test fails, you changed the descriptor format. Follow the
// BUMP RULE in llama-state-layout.h (update the goldens here; bump STATEOS_KV_LAYOUT_VERSION when old states must no
// longer load) — never "fix" the goldens without that decision.

#include "../src/llama-state-layout.h"

#include <cstdio>
#include <string>

static int g_failures = 0;
static int g_checks   = 0;

static void check_eq(const std::string & got, const std::string & want, const char * what) {
    ++g_checks;
    if (got != want) {
        ++g_failures;
        std::fprintf(stderr, "FAIL %s\n  got:  %s\n  want: %s\n", what, got.c_str(), want.c_str());
    }
}

static void check(bool ok, const char * what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL %s\n", what);
    }
}

// a qwen4exp-shaped context: hybrid (recurrent + attention), DSA indexer, flash attention, q8_0 K/V
static llama_state_layout_info hybrid() {
    llama_state_layout_info L;
    L.rope_type = 8; L.rope_freq_base = 10000000.0f; L.rope_freq_scale = 1.0f; L.n_ctx_orig_yarn = 262144;
    L.yarn_ext_factor = 0.0f; L.yarn_attn_factor = 1.0f; L.yarn_beta_fast = 32.0f; L.yarn_beta_slow = 1.0f;
    L.arch = "qwen4exp"; L.seq_version = 4; L.n_ctx = 196608; L.kv_size = 196608; L.v_state = 0; L.n_layer = 4;
    L.flash_attn = 1; L.mla_attn = 0; L.k_hadamard = 0; L.v_hadamard = 0; L.idx_hadamard = 1; L.compacted = false;
    L.k_rows = { { 3, 8, 544 } };
    L.v_rows = { { 3, 8, 512 } };
    L.qnext  = 1;
    L.s_rows = { { 0, 0, 12345 }, { 1, 0, 12345 }, { 2, 0, 12345 } };
    L.has_indexer = 1;
    L.r_rows = { { 3, 1, 128 } };
    return L;
}

int main() {
    check_eq(llama_state_layout_render(hybrid()),
             "rope=type=8 base=10000000 scale=1 orig_yarn=262144 ext=0 attn=1 beta_fast=32 beta_slow=1\n"
             "kv=arch=qwen4exp seqv=4 n_ctx=196608 size=196608 v_state=0 n_layer=4 fa=1 mla=0 khad=0 vhad=0 ihad=1 compact=0"
             " k3=8/544 v3=8/512 qnext=1 s0=0/12345 s1=0/12345 s2=0/12345 idx=1 r3=1/128\n",
             "golden: hybrid qwen4exp-shaped context");

    // compacted SWA, transposed V with one layer lacking V, no recurrent state, no indexer, non-integral rope values
    llama_state_layout_info L;
    L.rope_type = 2; L.rope_freq_base = 1000000.5f; L.rope_freq_scale = 0.25f; L.n_ctx_orig_yarn = 32768;
    L.yarn_ext_factor = -1.0f; L.yarn_attn_factor = 1.0f; L.yarn_beta_fast = 32.0f; L.yarn_beta_slow = 1.0f;
    L.arch = "llama"; L.seq_version = 4; L.n_ctx = 8192; L.kv_size = 8192; L.v_state = 1; L.n_layer = 2;
    L.flash_attn = 0; L.mla_attn = 0; L.k_hadamard = 1; L.v_hadamard = 0; L.idx_hadamard = 1;
    L.compacted = true; L.size_swa = 4096; L.sink_rows = 4;
    L.k_rows = { { 0, 1, 256 }, { 1, 1, 256 } };
    L.v_rows = { { 1, 1, 128 } };
    L.qnext = 0; L.has_indexer = 0;
    check_eq(llama_state_layout_render(L),
             "rope=type=2 base=1000000.5 scale=0.25 orig_yarn=32768 ext=-1 attn=1 beta_fast=32 beta_slow=1\n"
             "kv=arch=llama seqv=4 n_ctx=8192 size=8192 v_state=1 n_layer=2 fa=0 mla=0 khad=1 vhad=0 ihad=1 compact=1"
             " size_swa=4096 sink_rows=4 k0=1/256 k1=1/256 v1=1/128 qnext=0 idx=0\n",
             "golden: compacted, transposed V, partial V");

    // no V cache at all (v_state 2): V rows are never printed
    L.v_state = 2;
    const std::string no_v = llama_state_layout_render(L);
    check(no_v.find(" v1=") == std::string::npos, "v_state 2 prints no V rows");

    // every field reaches the text (a digest over it then separates the layouts)
    llama_state_layout_info a = hybrid();
    const std::string base = llama_state_layout_render(a);
    check(llama_state_layout_render(hybrid()) == base, "deterministic");
    a.k_rows[0].type = 1;          check(llama_state_layout_render(a) != base, "K type");          a = hybrid();
    a.v_rows[0].width = 256;       check(llama_state_layout_render(a) != base, "V width");         a = hybrid();
    a.k_hadamard = 1;              check(llama_state_layout_render(a) != base, "K Hadamard");      a = hybrid();
    a.n_ctx = 131072;              check(llama_state_layout_render(a) != base, "n_ctx");           a = hybrid();
    a.s_rows.pop_back();           check(llama_state_layout_render(a) != base, "recurrent rows");  a = hybrid();
    a.r_rows[0].type = 0;          check(llama_state_layout_render(a) != base, "indexer type");    a = hybrid();
    a.rope_freq_base = 5000000.0f; check(llama_state_layout_render(a) != base, "rope base");

    std::printf("test-stateos-layout: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
