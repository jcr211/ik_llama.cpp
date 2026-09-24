#pragma once

// qwen4exp per-layer-embedding (PLE) n-gram history.
//
// Each token's PLE rows hash the token with its n_gram - 1 predecessors. Predecessors that are not
// in the ubatch come from a per-sequence history kept on the host (llama_context::ple_hist). These
// helpers are the whole history semantics, shared by the input builder (llama.cpp), the
// llama_ple_history_* API and tests/test-ple-hist.cpp.
//
// H is any type with `llama_pos next_pos` and `std::vector<llama_token> toks`.

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

// LONGSPEAR_PLE_HIST_REWIND=1: keep the history exact across rewinds. Unset: base behaviour.
static inline bool llama_ple_hist_rewind_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LONGSPEAR_PLE_HIST_REWIND");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

// LONGSPEAR_PLE_HIST_LOG=1: one stderr line per mid-sequence EOS reset. Not every one is a rewind
// defect: an M-RoPE image puts all its positions at pos_0, so an image longer than one ubatch
// resets on each ubatch after the first (vision mode only; text-only runs expect zero).
static inline bool llama_ple_hist_log_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LONGSPEAR_PLE_HIST_LOG");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

// The id the builder hashes for a position without a token id (an embedding ubatch: image
// patches): the model's image token, EOS when it names none.
static inline llama_token llama_ple_media_token(llama_token eos, uint32_t image_token_id) {
    return image_token_id != 0 ? (llama_token) image_token_id : eos;
}

// The history a sequential decode holds before `next_pos` after decoding `prev`: its last
// n_gram - 1 tokens, front-padded with EOS when there are fewer, which at next_pos == n_prev is
// exactly the position-0 convention. A negative id (LLAMA_TOKEN_NULL, a media position in the
// server's cache tokens) becomes `media`, the id the builder pushed for that position: text after
// an image continues at next_pos (M-RoPE images take one position), so no reset intervenes.
template <typename H>
static inline void llama_ple_hist_assign(H & h, int32_t n_gram, llama_token eos, llama_token media,
        const llama_token * prev, int32_t n_prev, llama_pos next_pos) {
    const int32_t keep = n_gram - 1;
    const int32_t n    = prev != nullptr ? std::min(std::max(n_prev, 0), keep) : 0;

    h.toks.assign(keep, eos);
    for (int32_t i = 0; i < n; ++i) {
        const llama_token tok = prev[n_prev - n + i];
        h.toks[keep - n + i] = tok < 0 ? media : tok;
    }
    h.next_pos = next_pos;
}

// Speculative resume (common_speculative_ple_resume): from the history snapshot taken at the
// checkpoint, the tokens before the resume position and that position. A replay restore resumes at
// n_past with the snapshot itself; a per-step (direct) restore resumes after the sampled token and
// the accepted drafts ids[0 .. n-2], at n_past + n. False when the snapshot is not contiguous with
// the checkpoint, so there is nothing exact to rebuild from.
static inline bool llama_ple_hist_spec_resume(
        const std::vector<llama_token> & snap, llama_pos snap_next_pos, llama_pos n_past,
        llama_token sampled, const std::vector<llama_token> & ids, bool direct,
        std::vector<llama_token> & prev, llama_pos & next_pos) {
    if (snap_next_pos != n_past || ids.empty()) {
        return false;
    }
    prev     = snap;
    next_pos = n_past;
    if (direct) {
        prev.push_back(sampled);
        prev.insert(prev.end(), ids.begin(), ids.end() - 1);
        next_pos += (llama_pos) ids.size();
    }
    return true;
}

// Server prompt resume: the last n_hist tokens before the first decoded position, which follows
// the system tokens and then cache[0 .. n_past) (the server's cache tokens, LLAMA_TOKEN_NULL at
// media positions).
template <typename Cache>
static inline std::vector<llama_token> llama_ple_hist_prompt_window(
        const std::vector<llama_token> & system_tokens, const Cache & cache, int32_t n_past, int32_t n_hist) {
    std::vector<llama_token> prev;
    for (int32_t i = std::max(0, (int32_t) system_tokens.size() - n_hist); i < (int32_t) system_tokens.size(); ++i) {
        prev.push_back(system_tokens[i]);
    }
    for (int32_t i = std::max(0, n_past - n_hist); i < n_past; ++i) {
        prev.push_back(cache[i]);
    }
    return prev;
}

// Writes the n-gram context of every ubatch token into ctx (n_tokens x n_gram: [i*n_gram] is the
// token, [i*n_gram + s] its s-th predecessor) and advances each sequence's history past the ubatch.
// A sequence whose history is not contiguous with its first position in the ubatch is reset to
// EOS; on_reset(seq, pos, next_pos) is called for each such reset.
template <typename H, typename OnReset>
static inline void llama_ple_ngram_fill(
        std::map<llama_seq_id, H> & hist_map,
        int32_t                     n_tokens,
        const llama_token         * token,     // nullptr for an embedding ubatch
        llama_token                 img_tok,   // stands in for every position of an embedding ubatch
        const llama_pos           * pos,
        llama_seq_id * const      * seq_id,
        int32_t                     n_gram,
        llama_token                 eos,
        std::vector<llama_token>  & ctx,
        OnReset                  && on_reset) {
    auto tok_of = [&](int32_t k) -> llama_token {
        return token ? token[k] : img_tok;
    };

    // snapshot before any update: one pass would let a token read an earlier token of this
    // same ubatch as prior context
    std::map<llama_seq_id, std::vector<llama_token>> snap;
    for (int32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id seq = seq_id[i][0];
        if (snap.count(seq)) {
            continue;
        }
        auto & h = hist_map[seq];
        if (h.next_pos != pos[i]) {
            on_reset(seq, pos[i], h.next_pos);
            h.toks.assign(n_gram - 1, eos);
        }
        h.toks.resize(n_gram - 1, eos);
        snap[seq] = h.toks;
    }

    ctx.resize((size_t) n_tokens * n_gram);

    for (int32_t i = 0; i < n_tokens; ++i) {
        const llama_pos    p   = pos[i];
        const llama_seq_id seq = seq_id[i][0];

        const auto & hist = snap[seq];

        // predecessor s (1-based) of this token: from the ubatch when it is there, from the
        // sequence's own history when it is not, EOS past a segment boundary
        auto prev = [&](int32_t s) -> llama_token {
            const int32_t j = i - s;
            if (j >= 0 && seq_id[j][0] == seq && pos[j] == p - s) {
                return tok_of(j);
            }
            // s - i positions before this ubatch started, most recent last
            const int32_t back = s - i;
            const int32_t k    = (int32_t) hist.size() - back;
            if (back > 0 && k >= 0 && p - s >= 0) {
                return hist[k];
            }
            return eos;
        };

        llama_token * ctx_toks = ctx.data() + (size_t) i * n_gram;
        ctx_toks[0] = tok_of(i);
        bool cut = false;
        for (int32_t s = 1; s < n_gram; ++s) {
            ctx_toks[s] = cut ? eos : prev(s);
            if (ctx_toks[s] == eos) {
                cut = true;
            }
        }

        auto & h = hist_map[seq];
        h.toks.push_back(tok_of(i));
        if ((int32_t) h.toks.size() > n_gram - 1) {
            h.toks.erase(h.toks.begin(), h.toks.end() - (n_gram - 1));
        }
        h.next_pos = p + 1;
    }
}
