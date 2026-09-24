// Synthetic (no model) tests of the qwen4exp PLE n-gram history (src/llama-ple-hist.h): the
// setter's padding and truncation, and the input builder's n-gram contexts after a rewind compared
// with a sequential decode of the same tokens. The PLE rows are a pure hash of these contexts, so
// equal contexts mean equal rows.

#include "../src/llama-ple-hist.h"

#include <cstdio>
#include <map>
#include <vector>

namespace {

constexpr int32_t     N_GRAM = 3;      // qwen4exp ple.ngram_size
constexpr llama_token EOS    = 151645;
constexpr llama_token IMG    = 151655;
constexpr llama_seq_id SEQ   = 0;

int n_fail = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
            ++n_fail;                                                            \
        }                                                                        \
    } while (0)

struct hist {
    llama_pos                next_pos = -1;
    std::vector<llama_token> toks;
};

using tokens     = std::vector<llama_token>;
using ctx_by_pos = std::map<llama_pos, tokens>; // n-gram context of the latest decode at each position

// drives llama_ple_ngram_fill the way llama_set_inputs does
struct decoder {
    std::map<llama_seq_id, hist> m;
    int n_mid_resets = 0; // resets at pos > 0: what LONGSPEAR_PLE_HIST_LOG reports

    void decode(llama_seq_id seq, const tokens & toks, llama_pos pos0, ctx_by_pos & out,
            int32_t n_ubatch = 512, bool embd = false) {
        for (int32_t b = 0; b < (int32_t) toks.size(); b += n_ubatch) {
            const int32_t n = std::min(n_ubatch, (int32_t) toks.size() - b);
            tokens tk(toks.begin() + b, toks.begin() + b + n);
            std::vector<llama_pos>      pos(n);
            std::vector<llama_seq_id>   seqs(n, seq);
            std::vector<llama_seq_id *> seq_ptr(n);
            for (int32_t i = 0; i < n; ++i) {
                pos[i]     = pos0 + b + i;
                seq_ptr[i] = &seqs[i];
            }
            tokens ctx;
            llama_ple_ngram_fill(m, n, embd ? nullptr : tk.data(), IMG, pos.data(), seq_ptr.data(),
                    N_GRAM, EOS, ctx, [&](llama_seq_id, llama_pos p, llama_pos) {
                        if (p > 0) {
                            ++n_mid_resets;
                        }
                    });
            for (int32_t i = 0; i < n; ++i) {
                out[pos[i]] = tokens(ctx.begin() + i * N_GRAM, ctx.begin() + (i + 1) * N_GRAM);
            }
        }
    }

    // llama_ple_history_set
    void set(llama_seq_id seq, const tokens & prev, llama_pos next_pos) {
        llama_ple_hist_assign(m[seq], N_GRAM, EOS, prev.data(), (int32_t) prev.size(), next_pos);
    }
};

tokens make_tokens(int n, llama_token base) {
    tokens t(n);
    for (int i = 0; i < n; ++i) {
        t[i] = base + 7 * i;
    }
    return t;
}

tokens slice(const tokens & t, int a, int b) {
    return tokens(t.begin() + a, t.begin() + b);
}

bool same_from(const ctx_by_pos & a, const ctx_by_pos & b, llama_pos from) {
    for (const auto & [p, c] : a) {
        if (p < from) {
            continue;
        }
        const auto it = b.find(p);
        if (it == b.end() || it->second != c) {
            return false;
        }
    }
    return true;
}

void test_setter() {
    hist h;

    llama_ple_hist_assign(h, N_GRAM, EOS, nullptr, 0, 0);
    CHECK((h.toks == tokens{EOS, EOS}) && h.next_pos == 0);

    const tokens one = {7};
    llama_ple_hist_assign(h, N_GRAM, EOS, one.data(), 1, 1);
    CHECK((h.toks == tokens{EOS, 7}) && h.next_pos == 1);

    const tokens two = {7, 8};
    llama_ple_hist_assign(h, N_GRAM, EOS, two.data(), 2, 2);
    CHECK((h.toks == tokens{7, 8}) && h.next_pos == 2);

    const tokens many = {5, 6, 7, 8};
    llama_ple_hist_assign(h, N_GRAM, EOS, many.data(), 4, 1234);
    CHECK((h.toks == tokens{7, 8}) && h.next_pos == 1234);

    const tokens media = {7, LLAMA_TOKEN_NULL};
    llama_ple_hist_assign(h, N_GRAM, EOS, media.data(), 2, 40);
    CHECK((h.toks == tokens{7, EOS}) && h.next_pos == 40);

    llama_ple_hist_assign(h, N_GRAM, EOS, nullptr, 3, 5);
    CHECK((h.toks == tokens{EOS, EOS}) && h.next_pos == 5);

    llama_ple_hist_assign(h, N_GRAM, EOS, two.data(), -1, 6);
    CHECK((h.toks == tokens{EOS, EOS}) && h.next_pos == 6);

    // a longer n-gram keeps more predecessors
    llama_ple_hist_assign(h, 5, EOS, two.data(), 2, 2);
    CHECK((h.toks == tokens{EOS, EOS, 7, 8}) && h.next_pos == 2);
}

// rewinding to k and re-decoding: with the history set from tokens[0, k) the contexts match a
// sequential decode; without it the first two tokens after the rewind hash with EOS
void test_rewind_every_position() {
    const int    n = 12;
    const tokens t = make_tokens(n, 1000);

    ctx_by_pos ref;
    {
        decoder d;
        d.decode(SEQ, t, 0, ref);
        CHECK(d.n_mid_resets == 0);

        // the same contexts across ubatch splits
        decoder d3;
        ctx_by_pos ref3;
        d3.decode(SEQ, t, 0, ref3, 3);
        CHECK(ref3 == ref);
        CHECK(d3.n_mid_resets == 0);
    }

    for (int k = 0; k < n; ++k) {
        for (const bool fix : {false, true}) {
            decoder    d;
            ctx_by_pos out;
            d.decode(SEQ, slice(t, 0, k), 0, out);
            d.decode(SEQ, make_tokens(4, 9000), k, out); // tokens later rejected
            const int resets_before = d.n_mid_resets;
            if (fix) {
                d.set(SEQ, slice(t, 0, k), k);
            }
            d.decode(SEQ, slice(t, k, n), k, out);

            if (fix || k == 0) {
                CHECK(same_from(ref, out, 0));
                CHECK(d.n_mid_resets == resets_before);
            } else {
                // the defect: position k (and k + 1 when it has two true predecessors) differ
                CHECK(out[k] != ref[k]);
                if (k + 1 < n) {
                    CHECK(out[k + 1] != ref[k + 1]);
                }
                CHECK(same_from(ref, out, k + 2));
                CHECK(d.n_mid_resets == resets_before + 1);
            }
        }
    }
}

// common_speculative_ple_resume: history snapshot at the checkpoint (n_past), then either a replay
// from n_past (gpu-fallback / cpu) or a direct per-step restore after `step` accepted drafts
void test_speculative(bool direct, int step) {
    const int      n     = 16;
    const int      K     = 6;     // ckpt.n_past: position of the sampled token
    const int      n_dft = 4;
    const tokens   t     = make_tokens(n, 2000);

    ctx_by_pos ref;
    {
        decoder d;
        d.decode(SEQ, t, 0, ref);
    }

    for (const bool fix : {false, true}) {
        decoder    d;
        ctx_by_pos out;
        d.decode(SEQ, slice(t, 0, K), 0, out);

        // common_speculative_ple_snapshot (llama_ple_history_get)
        const hist snap = d.m[SEQ];
        CHECK(snap.next_pos == K);

        // verify batch [sampled, drafts]: the first `step` drafts are right, the rest wrong
        tokens verify = { t[K] };
        for (int j = 0; j < n_dft; ++j) {
            verify.push_back(j < step ? t[K + 1 + j] : 9000 + j);
        }
        d.decode(SEQ, verify, K, out);
        const int resets_before = d.n_mid_resets;

        // ids: the accepted drafts plus the new sampled token
        const tokens ids = slice(t, K + 1, K + 2 + step);

        if (fix) {
            tokens    prev     = snap.toks;
            llama_pos next_pos = K;
            if (direct) {
                prev.push_back(t[K]);
                prev.insert(prev.end(), ids.begin(), ids.end() - 1);
                next_pos += (llama_pos) ids.size();
            }
            d.set(SEQ, prev, next_pos);
        }

        llama_pos resume = K + 1 + step; // where the next verify batch starts
        if (!direct) {
            // replay the sampled token and the accepted drafts from the checkpoint
            tokens replay = { t[K] };
            replay.insert(replay.end(), ids.begin(), ids.end() - 1);
            d.decode(SEQ, replay, K, out);
        }
        d.decode(SEQ, slice(t, resume, n), resume, out);

        if (fix) {
            CHECK(same_from(ref, out, 0));
            CHECK(d.n_mid_resets == resets_before);
        } else {
            CHECK(!same_from(ref, out, 0));
            CHECK(d.n_mid_resets > resets_before);
        }
    }
}

// a rewound sequence does not disturb another sequence's history
void test_other_sequence_untouched() {
    const tokens t0 = make_tokens(8, 3000);
    const tokens t1 = make_tokens(8, 4000);

    decoder    d;
    ctx_by_pos out0;
    ctx_by_pos out1;
    d.decode(0, t0, 0, out0);
    d.decode(1, t1, 0, out1);
    const hist before = d.m[0];

    d.set(1, slice(t1, 0, 3), 3);
    CHECK(d.m[0].toks == before.toks && d.m[0].next_pos == before.next_pos);
    CHECK((d.m[1].toks == tokens{t1[1], t1[2]}) && d.m[1].next_pos == 3);
}

// an image (embedding ubatch) followed by text after a position jump: a sequential decode resets to
// EOS at the text; a rewind whose predecessors include the media positions (LLAMA_TOKEN_NULL in the
// server's cache tokens) reproduces it
void test_media_boundary() {
    const tokens pre  = make_tokens(4, 5000);
    const int    n_im = 5;
    const tokens text = make_tokens(6, 6000);
    const llama_pos p_img  = (llama_pos) pre.size();
    const llama_pos p_text = p_img + n_im + 3; // IMROPE: the text resumes past the image's positions

    // the server's cache tokens: text, then LLAMA_TOKEN_NULL for every media position, then text
    tokens cache = pre;
    cache.insert(cache.end(), n_im, LLAMA_TOKEN_NULL);
    cache.insert(cache.end(), text.begin(), text.end());

    ctx_by_pos ref;
    {
        decoder d;
        d.decode(SEQ, pre, 0, ref);
        d.decode(SEQ, tokens(n_im, 0), p_img, ref, 512, true);
        d.decode(SEQ, text, p_text, ref);
        CHECK(d.n_mid_resets == 1); // the jump to the text
    }

    for (int r = 0; r < (int) text.size(); ++r) {
        decoder    d;
        ctx_by_pos out;
        d.decode(SEQ, pre, 0, out);
        d.decode(SEQ, tokens(n_im, 0), p_img, out, 512, true);
        d.decode(SEQ, slice(text, 0, r), p_text, out);
        d.decode(SEQ, make_tokens(3, 9500), p_text + r, out); // rejected
        const int idx = (int) pre.size() + n_im + r;           // resume index in cache
        d.set(SEQ, slice(cache, 0, idx), p_text + r);
        d.decode(SEQ, slice(text, r, (int) text.size()), p_text + r, out);
        CHECK(same_from(ref, out, p_text));
    }
}

} // namespace

int main() {
    test_setter();
    test_rewind_every_position();
    for (const bool direct : {false, true}) {
        for (int step = 0; step <= 3; ++step) {
            test_speculative(direct, step);
        }
    }
    test_other_sequence_untouched();
    test_media_boundary();

    if (n_fail != 0) {
        fprintf(stderr, "test-ple-hist: %d check(s) failed\n", n_fail);
        return 1;
    }
    printf("test-ple-hist: all checks passed\n");
    return 0;
}
