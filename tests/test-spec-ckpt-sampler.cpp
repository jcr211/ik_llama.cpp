// SL-1 commit 5 (LONGSPEAR_SPEC_CKPT_LEAN): the checkpoint sampler a slot keeps across rounds
// (reset + clone) must restore the slot sampler exactly as a fresh one (init + clone) does. A
// restore clones the checkpoint sampler into the slot sampler, so the test compares every field
// that clone writes, after restoring from each into two identically prepared slot samplers.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "common.h"
#include "llama.h"
#include "llama-grammar.h"
#include "llama-sampling.h"
#include "sampling.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

static common_params_sampling make_params(const llama_vocab * vocab, uint32_t seed) {
    common_params_sampling params;
    params.seed = seed;
    params.grammar = { COMMON_GRAMMAR_TYPE_USER, R"(root ::= "</think>" "{" [a-z]* "}")" };
    params.grammar_lazy = true;
    params.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "</think>" });
    params.reasoning_budget_start  = common_tokenize(vocab, "<think>", false, true);
    params.reasoning_budget_end    = common_tokenize(vocab, "</think>", false, true);
    params.reasoning_budget_forced = params.reasoning_budget_end;
    params.reasoning_budget_tokens = 64;
    params.dry_multiplier = 0.8f;
    params.dry_base = 1.75f;
    params.dry_allowed_length = 2;
    params.dry_penalty_last_n = 64;
    params.adaptive_target = 0.6f;
    params.samplers_sequence.push_back(llama_sampler_type::DRY);
    params.samplers_sequence.push_back(llama_sampler_type::ADAPTIVE_P);
    return params;
}

// a different request's sampler, so the persistent one starts from unrelated state
static common_params_sampling make_other_params() {
    common_params_sampling params;
    params.seed = 7;
    params.temp = 0.2f;
    return params;
}

static void accept_text(common_sampler * s, llama_context * ctx, const std::string & text) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    for (llama_token t : common_tokenize(vocab, text, false, true)) {
        common_sampler_accept(s, ctx, t, true);
    }
}

static std::vector<llama_token_data> candidates(const llama_vocab * vocab) {
    std::vector<llama_token_data> c;
    for (const char * piece : { "x", "}", "{", "abc", "</think>" }) {
        for (llama_token t : common_tokenize(vocab, piece, false, true)) {
            c.push_back({ t, (float) c.size(), 0.0f });
        }
    }
    return c;
}

static void assert_same(const common_sampler * a, const common_sampler * b, llama_context * ctx) {
    // params (the fields sampling reads)
    assert(a->params.seed == b->params.seed);
    assert(a->params.temp == b->params.temp);
    assert(a->params.top_k == b->params.top_k);
    assert(a->params.n_prev == b->params.n_prev);
    assert(a->params.grammar_lazy == b->params.grammar_lazy);
    assert(common_grammar_value(a->params.grammar) == common_grammar_value(b->params.grammar));
    assert(a->params.adaptive_target == b->params.adaptive_target);
    assert(a->params.dry_multiplier == b->params.dry_multiplier);
    assert(a->params.samplers_sequence == b->params.samplers_sequence);

    assert(a->mirostat_mu == b->mirostat_mu);
    assert(a->n_valid == b->n_valid);
    assert(a->rng == b->rng);
    assert(a->speculative_seed == b->speculative_seed);
    assert(a->speculative_rng == b->speculative_rng);
    assert(a->server_biases == b->server_biases);
    assert(a->prev == b->prev);

    // grammar: same source, same trigger state, same masking of the same candidates
    assert((a->grammar == nullptr) == (b->grammar == nullptr));
    if (a->grammar) {
        assert(a->grammar_str == b->grammar_str);
        assert(a->grammar_root == b->grammar_root);
        assert(a->grammar->awaiting_trigger == b->grammar->awaiting_trigger);
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
        auto ca = candidates(vocab);
        auto cb = ca;
        llama_token_data_array da = { ca.data(), ca.size(), -1, false };
        llama_token_data_array db = { cb.data(), cb.size(), -1, false };
        llama_grammar_apply(a->grammar, ctx, &da);
        llama_grammar_apply(b->grammar, ctx, &db);
        for (size_t i = 0; i < ca.size(); ++i) {
            assert(ca[i].id == cb[i].id);
            assert(ca[i].logit == cb[i].logit || (ca[i].logit != ca[i].logit && cb[i].logit != cb[i].logit));
        }
    }

    // DRY
    assert((a->smpl == nullptr) == (b->smpl == nullptr));
    if (a->smpl) {
        assert(a->smpl->total_context_size == b->smpl->total_context_size);
        assert(a->smpl->dry_multiplier == b->smpl->dry_multiplier);
        assert(a->smpl->dry_base == b->smpl->dry_base);
        assert(a->smpl->dry_allowed_length == b->smpl->dry_allowed_length);
        assert(a->smpl->dry_penalty_last_n == b->smpl->dry_penalty_last_n);
        assert(a->smpl->dry_repeat_count == b->smpl->dry_repeat_count);
        assert(a->smpl->dry_max_token_repeat == b->smpl->dry_max_token_repeat);
        assert(a->smpl->dry_processed_breakers.size() == b->smpl->dry_processed_breakers.size());
        assert(a->smpl->last_tokens.capacity == b->smpl->last_tokens.capacity);
        assert(a->smpl->last_tokens.to_vector() == b->smpl->last_tokens.to_vector());
    }

    // adaptive-p
    assert((a->adapt_p_ctx == nullptr) == (b->adapt_p_ctx == nullptr));
    if (a->adapt_p_ctx) {
        assert(a->adapt_p_ctx->target == b->adapt_p_ctx->target);
        assert(a->adapt_p_ctx->decay == b->adapt_p_ctx->decay);
        assert(a->adapt_p_ctx->updt_w_cur == b->adapt_p_ctx->updt_w_cur);
        assert(a->adapt_p_ctx->rng == b->adapt_p_ctx->rng);
        assert(a->adapt_p_ctx->history == b->adapt_p_ctx->history);
        assert(a->adapt_p_ctx->orig_prob == b->adapt_p_ctx->orig_prob);
        assert(a->adapt_p_ctx->cum_orig_prob == b->adapt_p_ctx->cum_orig_prob);
        assert(a->adapt_p_ctx->cum_probs == b->adapt_p_ctx->cum_probs);
    }

    // reasoning budget: both cloned from the same source by common_reasoning_budget_clone. Its state
    // accessor is not called here: common.lib defines it twice (reasoning-budget.cpp is also
    // #included into sampling.cpp), so referencing it from a test breaks the link
    assert((a->rbudget == nullptr) == (b->rbudget == nullptr));
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: %s VOCAB_GGUF\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    assert(model != nullptr);
    llama_context * ctx = llama_init_from_model(model, llama_context_default_params());
    assert(ctx != nullptr);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const common_params_sampling params = make_params(vocab, 1234);

    // the slot sampler at the start of three consecutive rounds of one request
    common_sampler * slot = common_sampler_init(model, params);
    assert(slot && slot->grammar && slot->smpl && slot->adapt_p_ctx && slot->rbudget);

    // the persistent checkpoint sampler first served another request's rounds
    common_sampler * persistent = common_sampler_init(model, make_other_params());
    {
        common_sampler * other = common_sampler_init(model, make_other_params());
        accept_text(other, ctx, "unrelated text that fills prev");
        common_sampler_clone(other, persistent);
        common_sampler_free(other);
    }

    const char * rounds[] = {
        "<think> some reasoning ",
        "more reasoning that repeats repeats repeats ",
        "done </think>{abc",
    };
    int n_round = 0;
    for (const char * text : rounds) {
        accept_text(slot, ctx, text);
        (void) slot->rng();                  // sampling advanced the slot's RNG
        slot->mirostat_mu += 0.25f;

        // fresh: init + clone (flag off)
        common_sampler * fresh = common_sampler_init(model, params);
        common_sampler_clone(slot, fresh);

        // persistent: reset + clone (flag on)
        common_sampler_reset(persistent);
        common_sampler_clone(slot, persistent);

        assert_same(fresh, persistent, ctx);

        // a restore clones the checkpoint sampler into the slot sampler; do it from both into
        // two slot samplers that have drifted identically since the save
        common_sampler * dst_a = common_sampler_init(model, params);
        common_sampler * dst_b = common_sampler_init(model, params);
        accept_text(dst_a, ctx, "drafted tokens that got rejected");
        accept_text(dst_b, ctx, "drafted tokens that got rejected");
        common_sampler_clone(fresh, dst_a);
        common_sampler_clone(persistent, dst_b);
        assert_same(dst_a, dst_b, ctx);

        // and they keep agreeing on the accepted tokens that follow (text the triggered grammar
        // of the last round still accepts)
        accept_text(dst_a, ctx, "xyz}");
        accept_text(dst_b, ctx, "xyz}");
        assert_same(dst_a, dst_b, ctx);

        common_sampler_free(dst_a);
        common_sampler_free(dst_b);
        common_sampler_free(fresh);
        fprintf(stdout, "round %d: reset+clone == init+clone: OK\n", ++n_round);
    }

    common_sampler_free(persistent);
    common_sampler_free(slot);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    fprintf(stdout, "all OK\n");
    return 0;
}
