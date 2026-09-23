// SL-1 (tail-aware PER_STEP speculative checkpoints), CPU backend only.
//
// (a) the PLE-history slots a K-token verify batch saves through llama_ple_conv_save_per_step are
//     bit-equal to the history after running tokens 0..j one at a time, for K in {2..5, 17} and
//     every j; the batch's final history equals the sequential one; restoring slot j at
//     llama_ple_conv_row_offset rewrites only the row tail.
// (b) index convention: slot j of the delta-net per-step states, of the ssm_conv per-step states
//     and of the PLE-history slots all hold the state after verify token j.
//
// The graph code under test is src/llama-ple-conv.h, the same helpers build_qwen4exp.cpp uses.

#include "ggml.h"
#include "llama-ple-conv.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...) do {                                   \
    if (!(cond)) {                                              \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);    \
        fprintf(stderr, __VA_ARGS__);                           \
        fprintf(stderr, "\n");                                  \
        g_failures++;                                           \
    }                                                           \
} while (0)

static ggml_context * make_ctx(size_t mb) {
    ggml_init_params ip = { mb*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        fprintf(stderr, "ggml_init failed\n");
        exit(2);
    }
    return ctx;
}

static void fill(ggml_tensor * t, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    float * p = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        p[i] = d(rng);
    }
}

static void compute(ggml_context * ctx, ggml_cgraph * gf, int n_threads) {
    if (ggml_graph_compute_with_ctx(ctx, gf, n_threads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        exit(2);
    }
}

static bool bit_equal(const float * a, const float * b, size_t n) {
    return memcmp(a, b, n*sizeof(float)) == 0;
}

static double rel_l2(const float * a, const float * ref, size_t n) {
    double d2 = 0.0, r2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) a[i] - (double) ref[i];
        d2 += d*d;
        r2 += (double) ref[i]*ref[i];
    }
    return r2 > 0.0 ? std::sqrt(d2/r2) : std::sqrt(d2);
}

// PLE history as the qwen4exp graph keeps it: a state row [prefix | hist*hc_dim tail], the tail
// laid out [hist, hc_dim] (column t of channel c at c*hist + t)
struct ple_case {
    int32_t hist;
    int32_t hc_dim;
    int32_t prefix; // delta-net floats in front of the tail
};

// history after token t of xt ([n_tokens, hc_dim], token t of channel c at c*n_tokens + t), one
// token per graph exactly as a single-token decode builds it; out[t] = history after token t
static std::vector<std::vector<float>> ple_sequential(const ple_case & pc, const std::vector<float> & h0,
        const std::vector<float> & xt, int32_t n_tokens) {
    std::vector<std::vector<float>> out;
    std::vector<float> h = h0;
    for (int32_t t = 0; t < n_tokens; ++t) {
        ggml_context * ctx = make_ctx(16);
        ggml_tensor * state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pc.hist, pc.hc_dim);
        memcpy(state->data, h.data(), h.size()*sizeof(float));
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, pc.hc_dim);
        for (int32_t c = 0; c < pc.hc_dim; ++c) {
            ((float *) x->data)[c] = xt[(size_t) c*n_tokens + t];
        }
        ggml_tensor * conv_in = ggml_concat(ctx, state, x, 0);
        ggml_tensor * tail = ggml_cont(ctx, llama_ple_conv_window(ctx, conv_in, conv_in->ne[0] - pc.hist, pc.hist));
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, tail);
        compute(ctx, gf, 1);
        memcpy(h.data(), tail->data, h.size()*sizeof(float));
        out.push_back(h);
        ggml_free(ctx);
    }
    return out;
}

struct ple_batch_result {
    std::vector<std::vector<float>> slots; // per_step slot j
    std::vector<float> row;                // the state row after the batch
};

// one K-token batch the way qwen4exp_ple_conv builds it, with per-step slots
static ple_batch_result ple_batch(const ple_case & pc, const std::vector<float> & row0,
        const std::vector<float> & xt_data, int32_t n_tokens, int n_threads) {
    ggml_context * ctx = make_ctx(32);
    const int64_t ne0 = pc.prefix + (int64_t) pc.hist*pc.hc_dim;
    ggml_tensor * state_all = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 1);
    memcpy(state_all->data, row0.data(), row0.size()*sizeof(float));
    ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, pc.hc_dim);
    memcpy(xt->data, xt_data.data(), xt_data.size()*sizeof(float));
    ggml_tensor * per_step = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (n_tokens - 1) * pc.hist * pc.hc_dim);
    memset(per_step->data, 0, ggml_nbytes(per_step));

    const size_t esz     = ggml_element_size(state_all);
    const size_t row_off = llama_ple_conv_row_offset(state_all->ne[0], pc.hist, pc.hc_dim, esz);

    ggml_tensor * state = ggml_cont(ctx,
            ggml_view_2d(ctx, state_all, pc.hist, pc.hc_dim, pc.hist*esz, row_off));
    ggml_tensor * conv_in = ggml_concat(ctx, state, xt, 0);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    llama_ple_conv_save_per_step(ctx, gf, conv_in, per_step, pc.hist, pc.hc_dim, n_tokens);
    ggml_tensor * tail = ggml_cont(ctx, llama_ple_conv_window(ctx, conv_in, conv_in->ne[0] - pc.hist, pc.hist));
    ggml_tensor * dst  = ggml_view_2d(ctx, state_all, pc.hist, pc.hc_dim, pc.hist*esz, row_off);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, tail, dst));
    compute(ctx, gf, n_threads);

    ple_batch_result r;
    const size_t n_slot = (size_t) pc.hist*pc.hc_dim;
    for (int32_t j = 0; j + 1 < n_tokens; ++j) {
        const float * s = (const float *) per_step->data + (size_t) j*n_slot;
        r.slots.emplace_back(s, s + n_slot);
    }
    r.row.assign((const float *) state_all->data, (const float *) state_all->data + ne0);
    ggml_free(ctx);
    return r;
}

static void test_ple_per_step_matches_sequential() {
    std::mt19937 rng(20260924);
    const ple_case pc = { 9, 12, 7 };
    const size_t n_slot = (size_t) pc.hist*pc.hc_dim;

    for (int32_t K : { 2, 3, 4, 5, 17 }) {
        const int64_t ne0 = pc.prefix + (int64_t) n_slot;
        std::vector<float> row0(ne0);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (auto & v : row0) v = d(rng);
        std::vector<float> xt((size_t) K*pc.hc_dim);
        for (auto & v : xt) v = d(rng);

        const std::vector<float> h0(row0.begin() + pc.prefix, row0.end());
        const auto seq = ple_sequential(pc, h0, xt, K);

        for (int n_threads : { 1, 4 }) {
            const auto batch = ple_batch(pc, row0, xt, K, n_threads);
            CHECK((int32_t) batch.slots.size() == K - 1, "K=%d: %zu slots", K, batch.slots.size());
            for (int32_t j = 0; j + 1 < K; ++j) {
                CHECK(bit_equal(batch.slots[j].data(), seq[j].data(), n_slot),
                        "K=%d threads=%d: slot %d != history after token %d", K, n_threads, j, j);
                // an off-by-one would match the neighbour instead
                CHECK(!bit_equal(batch.slots[j].data(), seq[j + 1].data(), n_slot),
                        "K=%d: slot %d also matches token %d", K, j, j + 1);
            }
            CHECK(bit_equal(batch.row.data() + pc.prefix, seq[K - 1].data(), n_slot),
                    "K=%d threads=%d: final history != sequential", K, n_threads);
            CHECK(bit_equal(batch.row.data(), row0.data(), pc.prefix),
                    "K=%d threads=%d: the batch wrote into the delta-net prefix", K, n_threads);

            // restore slot j into a row whose tail went on to the end of the batch
            for (int32_t j = 0; j + 1 < K; ++j) {
                std::vector<float> row = batch.row;
                const size_t off = llama_ple_conv_row_offset(ne0, pc.hist, pc.hc_dim, sizeof(float));
                memcpy((char *) row.data() + off, batch.slots[j].data(), n_slot*sizeof(float));
                CHECK(bit_equal(row.data() + pc.prefix, seq[j].data(), n_slot),
                        "K=%d: restored slot %d != history after token %d", K, j, j);
                CHECK(bit_equal(row.data(), row0.data(), pc.prefix),
                        "K=%d: restoring slot %d touched the prefix", K, j);
            }
        }
    }
    fprintf(stdout, "test_ple_per_step_matches_sequential: %s\n", g_failures == 0 ? "OK" : "FAILED");
}

// ---------------------------------------------------------------------------------------------
// (b) index convention across the three per-step buffers

struct dn_dims {
    int64_t S;   // head dim (S_k == S_v)
    int64_t H;   // heads (H_k == H_v)
};

// delta_net over n_tokens tokens starting at token t0 of the full inputs; returns the final state
// and, when saved != nullptr, the per-step states
static std::vector<float> run_delta_net(const dn_dims & dd, const std::vector<float> & state0,
        const std::vector<float> & q, const std::vector<float> & k, const std::vector<float> & v,
        const std::vector<float> & g, const std::vector<float> & beta,
        int64_t n_all, int64_t t0, int64_t n_tokens, std::vector<float> * saved) {
    ggml_context * ctx = make_ctx(32);
    const int64_t S = dd.S, H = dd.H;
    ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_tokens, H, 1);
    ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_tokens, H, 1);
    ggml_tensor * tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_tokens, H, 1);
    ggml_tensor * tg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_tokens, 1, H, 1);
    ggml_tensor * tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, n_tokens, H, 1);
    ggml_tensor * ts = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S*H, 1, 1);
    // full inputs are laid out [S, n_all, H] / [n_all, H]; copy the token window
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t i = 0; i < S; ++i) {
                const size_t src = (size_t) h*n_all*S + (size_t) (t0 + t)*S + i;
                const size_t dst = (size_t) h*n_tokens*S + (size_t) t*S + i;
                ((float *) tq->data)[dst] = q[src];
                ((float *) tk->data)[dst] = k[src];
                ((float *) tv->data)[dst] = v[src];
            }
            ((float *) tg->data)[h*n_tokens + t] = g[(size_t) h*n_all + t0 + t];
            ((float *) tb->data)[h*n_tokens + t] = beta[(size_t) h*n_all + t0 + t];
        }
    }
    memcpy(ts->data, state0.data(), state0.size()*sizeof(float));

    const int64_t state_size = S*S*H;
    ggml_tensor * ts_saved = nullptr;
    if (saved) {
        ts_saved = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, std::max<int64_t>(1, (n_tokens - 1)*state_size));
        memset(ts_saved->data, 0, ggml_nbytes(ts_saved));
    }
    ggml_tensor * out = ggml_delta_net(ctx, tq, tk, tv, tg, tb, ts, ts_saved);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    compute(ctx, gf, 2);

    const int64_t output_size = S*H*n_tokens;
    std::vector<float> fin((const float *) out->data + output_size, (const float *) out->data + output_size + state_size);
    if (saved) {
        saved->assign((const float *) ts_saved->data, (const float *) ts_saved->data + (n_tokens - 1)*state_size);
    }
    ggml_free(ctx);
    return fin;
}

// ssm_conv over tokens [t0, t0 + n_tokens) of x ([d_inner, n_all]); returns the final conv state
// [d_conv - 1, d_inner] and, when saved != nullptr, the per-step states
static std::vector<float> run_ssm_conv(int64_t d_conv, int64_t d_inner, const std::vector<float> & s0,
        const std::vector<float> & x, const std::vector<float> & c, int64_t t0, int64_t n_tokens,
        std::vector<float> * saved) {
    ggml_context * ctx = make_ctx(16);
    ggml_tensor * ts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_conv - 1, d_inner, 1);
    memcpy(ts->data, s0.data(), s0.size()*sizeof(float));
    ggml_tensor * tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_inner, n_tokens);
    memcpy(tx->data, x.data() + (size_t) t0*d_inner, (size_t) n_tokens*d_inner*sizeof(float));
    ggml_tensor * tc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner);
    memcpy(tc->data, c.data(), c.size()*sizeof(float));
    ggml_tensor * tsq = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_tokens);
    memset(tsq->data, 0, ggml_nbytes(tsq));
    ggml_tensor * tsaved = nullptr;
    if (saved) {
        tsaved = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (d_conv - 1)*d_inner*n_tokens);
        memset(tsaved->data, 0, ggml_nbytes(tsaved));
    }
    ggml_tensor * out = ggml_ssm_conv(ctx, ts, tx, tc, tsq, tsaved);
    // the new state is the last d_conv - 1 columns of the [d_conv, d_inner] block after the outputs
    ggml_tensor * st = ggml_cont(ctx, ggml_view_2d(ctx, out, d_conv - 1, d_inner, d_conv*sizeof(float),
            (1 + d_inner*n_tokens)*sizeof(float)));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, st);
    compute(ctx, gf, 2);
    std::vector<float> fin((const float *) st->data, (const float *) st->data + (d_conv - 1)*d_inner);
    if (saved) {
        saved->assign((const float *) tsaved->data, (const float *) tsaved->data + (d_conv - 1)*d_inner*n_tokens);
    }
    ggml_free(ctx);
    return fin;
}

static void test_per_step_index_convention() {
    const int before = g_failures;
    std::mt19937 rng(424242);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    const dn_dims dd = { 16, 2 };
    const int64_t state_size = dd.S*dd.S*dd.H;
    const int64_t d_conv = 4, d_inner = 24;
    const ple_case pc = { 9, 12, 5 };
    const size_t n_ple = (size_t) pc.hist*pc.hc_dim;

    for (int64_t K : { 2, 3, 5, 17 }) {
        std::vector<float> q(dd.S*K*dd.H), k(dd.S*K*dd.H), v(dd.S*K*dd.H), g(K*dd.H), beta(K*dd.H);
        for (auto & x : q) x = d(rng);
        for (auto & x : k) x = d(rng);
        for (auto & x : v) x = d(rng);
        for (auto & x : g) x = -0.05f - 0.2f*std::fabs(d(rng));
        for (auto & x : beta) x = d(rng);
        std::vector<float> s0(state_size);
        for (auto & x : s0) x = 0.1f*d(rng);

        // delta-net: batch with per-step saves vs one token at a time
        std::vector<float> dn_saved;
        run_delta_net(dd, s0, q, k, v, g, beta, K, 0, K, &dn_saved);
        std::vector<std::vector<float>> dn_seq;
        std::vector<float> s = s0;
        for (int64_t t = 0; t < K; ++t) {
            s = run_delta_net(dd, s, q, k, v, g, beta, K, t, 1, nullptr);
            dn_seq.push_back(s);
        }

        // ssm_conv: batch with per-step saves vs one token at a time
        std::vector<float> xs(d_inner*K), cw(d_conv*d_inner), cs0((d_conv - 1)*d_inner);
        for (auto & x : xs) x = d(rng);
        for (auto & x : cw) x = d(rng);
        for (auto & x : cs0) x = d(rng);
        std::vector<float> conv_saved;
        run_ssm_conv(d_conv, d_inner, cs0, xs, cw, 0, K, &conv_saved);
        std::vector<std::vector<float>> conv_seq;
        std::vector<float> cs = cs0;
        for (int64_t t = 0; t < K; ++t) {
            cs = run_ssm_conv(d_conv, d_inner, cs, xs, cw, t, 1, nullptr);
            conv_seq.push_back(cs);
        }

        // PLE history
        std::vector<float> row0(pc.prefix + n_ple), xt((size_t) K*pc.hc_dim);
        for (auto & x : row0) x = d(rng);
        for (auto & x : xt) x = d(rng);
        const std::vector<float> h0(row0.begin() + pc.prefix, row0.end());
        const auto ple_seq = ple_sequential(pc, h0, xt, (int32_t) K);
        const auto ple_b   = ple_batch(pc, row0, xt, (int32_t) K, 2);

        const size_t conv_size = (size_t) (d_conv - 1)*d_inner;
        for (int64_t j = 0; j + 1 < K; ++j) {
            // delta-net slot j: nearest sequential state must be the one after token j
            const float * slot = dn_saved.data() + (size_t) j*state_size;
            int64_t best = -1;
            double best_rel = INFINITY;
            for (int64_t t = 0; t < K; ++t) {
                const double r = rel_l2(slot, dn_seq[t].data(), state_size);
                if (r < best_rel) {
                    best_rel = r;
                    best = t;
                }
            }
            CHECK(best == j && best_rel < 1e-5, "K=%lld: delta-net slot %lld matches token %lld (relL2 %.3e)",
                    (long long) K, (long long) j, (long long) best, best_rel);

            // ssm_conv slot j (the op keeps n_tokens slots; the restore reads the same index)
            CHECK(bit_equal(conv_saved.data() + (size_t) j*conv_size, conv_seq[j].data(), conv_size),
                    "K=%lld: conv slot %lld != conv state after token %lld", (long long) K, (long long) j, (long long) j);

            // PLE slot j
            CHECK(bit_equal(ple_b.slots[j].data(), ple_seq[j].data(), n_ple),
                    "K=%lld: PLE slot %lld != history after token %lld", (long long) K, (long long) j, (long long) j);
        }
    }
    fprintf(stdout, "test_per_step_index_convention: %s\n", g_failures == before ? "OK" : "FAILED");
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    test_ple_per_step_matches_sequential();
    test_per_step_index_convention();
    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    fprintf(stdout, "all OK\n");
    return 0;
}
