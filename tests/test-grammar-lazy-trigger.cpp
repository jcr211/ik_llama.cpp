#ifdef NDEBUG
#undef NDEBUG
#endif

#include "common.h"
#include "llama.h"
#include "llama-grammar.h"
#include "sampling.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static common_sampler * make_sampler(const llama_model * model, bool lazy, bool require_trigger) {
    common_params_sampling params;
    params.grammar = { COMMON_GRAMMAR_TYPE_USER, R"(root ::= "trigger")" };
    params.grammar_lazy = lazy;
    params.grammar_lazy_require_trigger = require_trigger;
    if (lazy) {
        params.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "trigger" });
    }

    common_sampler * sampler = common_sampler_init(model, params);
    assert(sampler != nullptr);
    assert(sampler->grammar != nullptr);
    assert(sampler->params.grammar_lazy_require_trigger == require_trigger);
    return sampler;
}

static std::vector<llama_token> find_eog_tokens(const llama_vocab * vocab) {
    std::vector<llama_token> result;
    for (llama_token token = 0; token < llama_vocab_n_tokens(vocab); ++token) {
        if (llama_vocab_is_eog(vocab, token)) {
            result.push_back(token);
        }
    }
    return result;
}

static std::vector<llama_token_data> make_candidates(
        const std::vector<llama_token> & eog_tokens,
        llama_token non_eog) {
    std::vector<llama_token_data> result;
    float logit = 1.0f;
    for (llama_token token : eog_tokens) {
        result.push_back({ token, logit++, 0.0f });
    }
    result.push_back({ non_eog, logit, 0.0f });
    return result;
}

static void apply_grammar(
        const common_sampler * sampler,
        llama_context * ctx,
        std::vector<llama_token_data> & candidates) {
    llama_token_data_array data = { candidates.data(), candidates.size(), -1, false };
    llama_grammar_apply(sampler->grammar, ctx, &data);
}

static void assert_eog_masked(
        const std::vector<llama_token_data> & candidates,
        size_t n_eog) {
    for (size_t i = 0; i < n_eog; ++i) {
        assert(std::isinf(candidates[i].logit));
        assert(candidates[i].logit < 0.0f);
    }
}

static void assert_logits_equal(
        const std::vector<llama_token_data> & actual,
        const std::vector<llama_token_data> & expected) {
    assert(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(actual[i].id == expected[i].id);
        assert(actual[i].logit == expected[i].logit);
    }
}

static void accept_text(common_sampler * sampler, llama_context * ctx, const std::string & text) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const std::vector<llama_token> tokens = common_tokenize(vocab, text, false, true);
    assert(!tokens.empty());
    for (llama_token token : tokens) {
        llama_grammar_accept_token(sampler->grammar, ctx, token);
    }
}

int main(int argc, char ** argv) {
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
    const std::vector<llama_token> eog_tokens = find_eog_tokens(vocab);
    assert(eog_tokens.size() >= 2);

    const std::vector<llama_token> non_eog_tokens = common_tokenize(vocab, "x", false, true);
    assert(!non_eog_tokens.empty());
    assert(!llama_vocab_is_eog(vocab, non_eog_tokens.front()));
    const llama_token non_eog = non_eog_tokens.front();

    {
        common_sampler * sampler = make_sampler(model, true, true);
        assert(sampler->grammar->lazy_require_trigger);
        assert(sampler->grammar->awaiting_trigger);
        fprintf(stdout, "test_lazy_require_trigger_reaches_sampler_params: OK\n");

        auto candidates = make_candidates(eog_tokens, non_eog);
        const float non_eog_logit = candidates.back().logit;
        apply_grammar(sampler, ctx, candidates);
        assert_eog_masked(candidates, eog_tokens.size());
        assert(candidates.back().logit == non_eog_logit);

        accept_text(sampler, ctx, "trigger");
        assert(!sampler->grammar->awaiting_trigger);

        candidates = make_candidates(eog_tokens, non_eog);
        const auto expected = candidates;
        apply_grammar(sampler, ctx, candidates);
        for (size_t i = 0; i < eog_tokens.size(); ++i) {
            assert(candidates[i].logit == expected[i].logit);
        }

        common_sampler_free(sampler);
        fprintf(stdout, "test_lazy_require_trigger_masks_all_eog_until_word_trigger: OK\n");
    }

    {
        common_sampler * sampler = make_sampler(model, true, false);
        assert(!sampler->grammar->lazy_require_trigger);

        auto candidates = make_candidates(eog_tokens, non_eog);
        const auto expected = candidates;
        apply_grammar(sampler, ctx, candidates);
        assert_logits_equal(candidates, expected);

        common_sampler_free(sampler);
        fprintf(stdout, "test_lazy_require_trigger_off_preserves_logits: OK\n");
    }

    {
        common_sampler * sampler = make_sampler(model, false, true);
        assert(!sampler->grammar->lazy);
        assert(!sampler->grammar->awaiting_trigger);

        accept_text(sampler, ctx, "trigger");
        auto candidates = make_candidates(eog_tokens, non_eog);
        const auto expected = candidates;
        apply_grammar(sampler, ctx, candidates);
        for (size_t i = 0; i < eog_tokens.size(); ++i) {
            assert(candidates[i].logit == expected[i].logit);
        }

        common_sampler_free(sampler);
        fprintf(stdout, "test_lazy_require_trigger_non_lazy_noop: OK\n");
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
