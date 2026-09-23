// SL-1 commit 0: the IQK MoE chunk partition returns early for a chunk that owns no row group.
// At the real expert shapes (down projection 640 -> 2560, MXFP4_R8) 24 threads give 24 chunks of
// ceil(320/24) = 14 row groups, so chunk 23 starts past the last group. The guard must not change
// any output: the MoE matmul (mul_mat_id and the fused up/gate op) is bit-identical at 20, 24 and
// 32 threads to the single-thread result, whose partition has no empty chunk.

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static ggml_context * make_ctx(size_t mb) {
    ggml_init_params ip = { mb*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        fprintf(stderr, "ggml_init failed\n");
        exit(2);
    }
    return ctx;
}

// n_as experts of [ne00, ne01], quantized row by row from uniform floats
static ggml_tensor * make_experts(ggml_context * ctx, std::mt19937 & rng, int64_t ne00, int64_t ne01, int64_t n_as) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4_R8, ne00, ne01, n_as);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> src(ne00*ne01);
    for (int64_t a = 0; a < n_as; ++a) {
        for (auto & v : src) v = d(rng);
        ggml_quantize_chunk(GGML_TYPE_MXFP4_R8, src.data(), (char *) t->data + a*t->nb[2], 0, ne01, ne00, nullptr, nullptr);
    }
    return t;
}

struct moe_case {
    const char * name;
    int64_t ne00;
    int64_t ne01;
    bool    fused;
};

static int run_case(const moe_case & mc, int64_t n_tokens) {
    const int64_t n_as = 8, n_used = 4;
    std::mt19937 rng((uint32_t) (mc.ne00*131 + mc.ne01*7 + n_tokens + (mc.fused ? 1 : 0)));

    std::vector<std::vector<float>> results;
    int failures = 0;
    for (int n_threads : { 1, 20, 24, 32 }) {
        // same seed per thread count, so the inputs are identical
        std::mt19937 rng_case = rng;
        ggml_context * ctx = make_ctx(512);
        ggml_tensor * up   = make_experts(ctx, rng_case, mc.ne00, mc.ne01, n_as);
        ggml_tensor * gate = mc.fused ? make_experts(ctx, rng_case, mc.ne00, mc.ne01, n_as) : nullptr;

        // the down projection reads one input row per selected expert, up/gate one shared row
        const int64_t b_rows = mc.fused ? 1 : n_used;
        ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, mc.ne00, b_rows, n_tokens);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (int64_t i = 0; i < ggml_nelements(b); ++i) {
            ((float *) b->data)[i] = d(rng_case);
        }

        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
        for (int64_t t = 0; t < n_tokens; ++t) {
            // distinct experts per token, varying across tokens
            for (int64_t k = 0; k < n_used; ++k) {
                ((int32_t *) ids->data)[t*n_used + k] = (int32_t) ((t*3 + k*2) % n_as);
            }
        }

        ggml_tensor * out = mc.fused
            ? ggml_moe_up_gate(ctx, up, gate, b, ids, GGML_UNARY_OP_SILU)
            : ggml_mul_mat_id(ctx, up, b, ids);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        if (ggml_graph_compute_with_ctx(ctx, gf, n_threads) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s T=%lld threads=%d: compute failed\n", mc.name, (long long) n_tokens, n_threads);
            ggml_free(ctx);
            return failures + 1;
        }
        results.emplace_back((const float *) out->data, (const float *) out->data + ggml_nelements(out));
        ggml_free(ctx);

        if (results.size() > 1) {
            const auto & ref = results.front();
            const auto & cur = results.back();
            if (cur.size() != ref.size() || memcmp(cur.data(), ref.data(), ref.size()*sizeof(float)) != 0) {
                size_t n_diff = 0;
                for (size_t i = 0; i < ref.size() && i < cur.size(); ++i) {
                    n_diff += memcmp(&ref[i], &cur[i], sizeof(float)) != 0;
                }
                fprintf(stderr, "FAIL %s T=%lld threads=%d: %zu of %zu outputs differ from 1 thread\n",
                        mc.name, (long long) n_tokens, n_threads, n_diff, ref.size());
                failures++;
            }
        }
    }
    fprintf(stdout, "%s T=%lld: %s\n", mc.name, (long long) n_tokens, failures ? "FAILED" : "bit-identical at 1/20/24/32 threads");
    return failures;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const moe_case cases[] = {
        { "mul_mat_id down 640->2560",  640, 2560, false }, // 24 threads: chunk 23 is empty
        { "mul_mat_id up 2560->640",   2560,  640, false },
        { "moe_up_gate 2560->640",     2560,  640, true  },
        { "moe_up_gate 640->2560",      640, 2560, true  }, // synthetic: an empty fused chunk at 24
    };
    int failures = 0;
    for (const auto & mc : cases) {
        for (int64_t n_tokens : { 1, 2, 5, 17 }) {
            failures += run_case(mc, n_tokens);
        }
    }
    if (failures) {
        fprintf(stderr, "%d case(s) failed\n", failures);
        return 1;
    }
    fprintf(stdout, "all OK\n");
    return 0;
}
