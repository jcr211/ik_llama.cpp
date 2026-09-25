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
            int32_t n_ubatch = 512) {
        decode_impl(seq, toks, pos0, out, n_ubatch, false);
    }

    // an M-RoPE image: an embedding ubatch whose first position section is pos_0 for every patch
    // (mtmd-helper set_position_mrope_2d); the text after it continues at pos_0 + 1 (n_pos = 1)
    void decode_image(llama_seq_id seq, int32_t n_patches, llama_pos pos_0, ctx_by_pos & out,
            int32_t n_ubatch = 512) {
        decode_impl(seq, tokens(n_patches, 0), pos_0, out, n_ubatch, true);
    }

    void decode_impl(llama_seq_id seq, const tokens & toks, llama_pos pos0, ctx_by_pos & out,
            int32_t n_ubatch, bool image) {
        for (int32_t b = 0; b < (int32_t) toks.size(); b += n_ubatch) {
            const int32_t n = std::min(n_ubatch, (int32_t) toks.size() - b);
            tokens tk(toks.begin() + b, toks.begin() + b + n);
            std::vector<llama_pos>      pos(n);
            std::vector<llama_seq_id>   seqs(n, seq);
            std::vector<llama_seq_id *> seq_ptr(n);
            for (int32_t i = 0; i < n; ++i) {
                pos[i]     = image ? pos0 : pos0 + b + i;
                seq_ptr[i] = &seqs[i];
            }
            tokens ctx;
            llama_ple_ngram_fill(m, n, image ? nullptr : tk.data(), IMG, pos.data(), seq_ptr.data(),
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

    // llama_ple_history_set (media: what LLAMA_TOKEN_NULL becomes; the model's image token)
    void set(llama_seq_id seq, const tokens & prev, llama_pos next_pos, llama_token media = IMG) {
        llama_ple_hist_assign(m[seq], N_GRAM, EOS, media, prev.data(), (int32_t) prev.size(), next_pos);
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

    llama_ple_hist_assign(h, N_GRAM, EOS, IMG, nullptr, 0, 0);
    CHECK((h.toks == tokens{EOS, EOS}) && h.next_pos == 0);

    const tokens one = {7};
    llama_ple_hist_assign(h, N_GRAM, EOS, IMG, one.data(), 1, 1);
    CHECK((h.toks == tokens{EOS, 7}) && h.next_pos == 1);

    const tokens two = {7, 8};
    llama_ple_hist_assign(h, N_GRAM, EOS, IMG, two.data(), 2, 2);
    CHECK((h.toks == tokens{7, 8}) && h.next_pos == 2);

    const tokens many = {5, 6, 7, 8};
    llama_ple_hist_assign(h, N_GRAM, EOS, IMG, many.data(), 4, 1234);
    CHECK((h.toks == tokens{7, 8}) && h.next_pos == 1234);

    // media positions become the image token the builder pushed for them
    const tokens media = {7, LLAMA_TOKEN_NULL};
    llama_ple_hist_assign(h, N_GRAM, EOS, IMG, media.data(), 2, 40);
    CHECK((h.toks == tokens{7, IMG}) && h.next_pos == 40);

    // ... and the builder's own media id matches it
    CHECK(llama_ple_media_token(EOS, IMG) == IMG);
    CHECK(llama_ple_media_token(EOS, 0) == EOS);

    llama_ple_hist_assign(h, N_GRAM, EOS, IMG, nullptr, 3, 5);
    CHECK((h.toks == tokens{EOS, EOS}) && h.next_pos == 5);

    llama_ple_hist_assign(h, N_GRAM, EOS, IMG, two.data(), -1, 6);
    CHECK((h.toks == tokens{EOS, EOS}) && h.next_pos == 6);

    // a longer n-gram keeps more predecessors
    llama_ple_hist_assign(h, 5, EOS, IMG, two.data(), 2, 2);
    CHECK((h.toks == tokens{EOS, EOS, 7, 8}) && h.next_pos == 2);
}

// the server prompt-resume window (llama_ple_hist_prompt_window at p0 = |system| + n_past): after
// any rewind to n_past the prompt continues exactly as a sequential decode of the same tokens
void test_prompt_resume() {
    const tokens t = make_tokens(9, 8000);

    for (const tokens & sys : { tokens{}, make_tokens(3, 7000) }) {
        tokens all = sys;
        all.insert(all.end(), t.begin(), t.end());

        ctx_by_pos ref;
        {
            decoder d;
            d.decode(SEQ, all, 0, ref);
        }

        for (int n_past = 0; n_past <= (int) t.size(); ++n_past) {
            const llama_pos p0 = (llama_pos) sys.size() + n_past;
            if (p0 == 0) {
                continue; // the server leaves position 0 to the builder
            }
            decoder    d;
            ctx_by_pos out;
            d.decode(SEQ, slice(all, 0, p0), 0, out);
            d.decode(SEQ, make_tokens(4, 9000), p0, out); // the suffix a checkpoint restore drops
            const int resets_before = d.n_mid_resets;

            d.set(SEQ, llama_ple_hist_prompt_window(sys, t, n_past, N_GRAM - 1), p0);
            d.decode(SEQ, slice(t, n_past, (int) t.size()), p0, out);

            CHECK(same_from(ref, out, 0));
            CHECK(d.n_mid_resets == resets_before);
        }
    }
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

        const llama_pos resume = K + 1 + step; // where the next verify batch starts

        if (fix) {
            tokens    prev;
            llama_pos next_pos = -1;
            CHECK(llama_ple_hist_spec_resume(snap.toks, snap.next_pos, K, t[K], ids, direct, prev, next_pos));
            CHECK(next_pos == (direct ? resume : K));
            d.set(SEQ, prev, next_pos);

            // nothing exact to rebuild from: a snapshot not at the checkpoint, or no ids
            tokens    unused;
            llama_pos unused_pos = -1;
            CHECK(!llama_ple_hist_spec_resume(snap.toks, K - 1, K, t[K], ids, direct, unused, unused_pos));
            CHECK(!llama_ple_hist_spec_resume(snap.toks, K, K, t[K], tokens{}, direct, unused, unused_pos));
        }

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

// an M-RoPE image followed by text: every patch sits at pos_0 and the text continues at pos_0 + 1,
// which is the history's next_pos, so a sequential decode does not reset and the text's first
// predecessors are the image token. A resume whose window holds media positions (LLAMA_TOKEN_NULL
// in the server's cache tokens, one per patch) must reproduce that.
void test_media_boundary() {
    const tokens    pre    = make_tokens(4, 5000);
    const int       n_im   = 5;
    const tokens    text   = make_tokens(6, 6000);
    const llama_pos p_img  = (llama_pos) pre.size();
    const llama_pos p_text = p_img + 1; // mtmd_image_tokens_get_n_pos() == 1 under M-RoPE

    tokens cache = pre;
    cache.insert(cache.end(), n_im, LLAMA_TOKEN_NULL);
    cache.insert(cache.end(), text.begin(), text.end());

    for (const int32_t n_ubatch : {512, 3}) {
        ctx_by_pos ref;
        {
            decoder d;
            d.decode(SEQ, pre, 0, ref);
            d.decode_image(SEQ, n_im, p_img, ref, n_ubatch);
            // one ubatch: no reset. Split: each ubatch after the first resets at pos_0 (base
            // builder behaviour; the vision-mode note on LONGSPEAR_PLE_HIST_LOG)
            CHECK(d.n_mid_resets == (n_ubatch >= n_im ? 0 : 1));
            // the last image ubatch holds >= 2 patches here, so the history is [IMG, IMG]
            CHECK((d.m[SEQ].toks == tokens{IMG, IMG}) && d.m[SEQ].next_pos == p_text);
            d.decode(SEQ, text, p_text, ref);
            CHECK((ref[p_text] == tokens{text[0], IMG, IMG}));
        }

        for (int r = 0; r < (int) text.size(); ++r) {
            for (const llama_token media : {IMG, EOS}) {
                decoder    d;
                ctx_by_pos out;
                d.decode(SEQ, pre, 0, out);
                d.decode_image(SEQ, n_im, p_img, out, n_ubatch);
                d.decode(SEQ, slice(text, 0, r), p_text, out);
                d.decode(SEQ, make_tokens(3, 9500), p_text + r, out); // rejected

                const int n_past = (int) pre.size() + n_im + r; // resume index in the cache tokens
                d.set(SEQ, llama_ple_hist_prompt_window(tokens{}, cache, n_past, N_GRAM - 1), p_text + r, media);
                d.decode(SEQ, slice(text, r, (int) text.size()), p_text + r, out);

                if (media == IMG) {
                    CHECK(same_from(ref, out, p_text));
                } else if (r < N_GRAM - 1) {
                    // round 1 mapped media to EOS: wrong within n_gram - 1 positions of the image
                    CHECK(!same_from(ref, out, p_text));
                }
            }
        }
    }
}

} // namespace

int main() {
    test_setter();
    test_rewind_every_position();
    test_prompt_resume();
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
