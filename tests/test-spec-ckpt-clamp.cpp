// SL-1 fix round (B2): the checkpoint-capacity clamp (LONGSPEAR_SPEC_CLAMP_TO_CKPT) verifies only the first
// M-1 tokens of a longer ngram-mod draft. ngram-mod counts a round with accepted/drafted < 0.5 as low
// acceptance and resets its table after three in a row, so it must divide by the verified length. A
// 16-token draft clamped to 4 whose 4 tokens are all accepted is a fully accepted round.
//
// Drives the public speculative API on a vocab-only context: an ngram-mod stage over a repeating
// token cycle drafts 16 tokens; three rounds accept 4. With common_speculative_truncate_draft (what the
// clamp calls) the table survives; without it (the pre-fix behaviour, kept as the control) it resets.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "common.h"
#include "llama.h"
#include "ngram-mod.h"
#include "speculative.h"

#include <cassert>
#include <cstdio>
#include <memory>
#include <vector>

// returns true when ngram-mod reset its table during the three rounds
static bool run_rounds(llama_context * ctx, bool truncate) {
    common_params_speculative params;
    params.n_max = 16;
    params.ngram_size_n = 4;

    common_speculative_stage_params ngram;
    ngram.type  = COMMON_SPECULATIVE_TYPE_NGRAM_MOD;
    ngram.n_max = 16;
    params.stages = { ngram };
    params.ngram_mod = std::make_shared<common_ngram_mod>(4, 1u << 16);

    common_speculative * spec = common_speculative_init(params, ctx);
    assert(spec != nullptr);

    // a 40-token cycle seen four times; the next token continues it
    const llama_token base = 1000;
    llama_tokens prompt;
    for (int i = 0; i < 160; ++i) {
        prompt.push_back(base + i % 40);
    }
    const llama_token id_last = base;

    common_speculative_begin(spec, prompt);
    assert(params.ngram_mod->get_used() > 0);

    const size_t capacity_drafts = 4; // LONGSPEAR_SPEC_CKPT_MAX_TOKENS=5 -> 4 drafts per verify
    for (int round = 0; round < 3; ++round) {
        const llama_tokens draft = common_speculative_draft(spec, params, prompt, id_last);
        assert(draft.size() == 16);
        for (size_t i = 0; i < draft.size(); ++i) {
            assert(draft[i] == base + (llama_token) ((i + 1) % 40));
        }
        if (truncate) {
            common_speculative_truncate_draft(spec, draft.size(), capacity_drafts);
        }
        common_speculative_accept(spec, (uint16_t) capacity_drafts); // every verified draft token accepted
    }

    const bool reset = params.ngram_mod->get_used() == 0;
    common_speculative_free(spec);
    return reset;
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

    const bool reset_without = run_rounds(ctx, false);
    fprintf(stdout, "control (no truncation): ngram_mod reset after 3 clamped rounds = %s\n", reset_without ? "yes" : "no");
    assert(reset_without); // the defect the clamp would cause without the truncation hook

    const bool reset_with = run_rounds(ctx, true);
    fprintf(stdout, "clamped draft truncated to 4: ngram_mod reset = %s\n", reset_with ? "yes" : "no");
    assert(!reset_with);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    fprintf(stdout, "all OK\n");
    return 0;
}
