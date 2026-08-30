#include "common.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdio>

static void test_composite_verify_capacity() {
    common_params_speculative params;

    common_speculative_stage_params ngram;
    ngram.type  = COMMON_SPECULATIVE_TYPE_NGRAM_MOD;
    ngram.n_min = 4;

    common_speculative_stage_params mtp;
    mtp.type  = COMMON_SPECULATIVE_TYPE_MTP;
    mtp.n_max = 4;

    params.stages = { ngram, mtp };

    // ngram inherits the flat default n_max=16. The target verifies one sampled root plus those
    // 16 drafts, so K=17 is expected and every verify/checkpoint capacity must account for it.
    assert(params.get_max_stage_n_max() == 16);
    assert(params.get_max_verify_batch_tokens() == 17);
    assert(params.get_min_usable_stage_n_min() == 0);
}

static void test_stage_override_controls_capacity() {
    common_params_speculative params;
    params.n_max = 31;

    common_speculative_stage_params ngram;
    ngram.type  = COMMON_SPECULATIVE_TYPE_NGRAM_MOD;
    ngram.n_min = 4;
    ngram.n_max = 7;

    common_speculative_stage_params mtp;
    mtp.type  = COMMON_SPECULATIVE_TYPE_MTP;
    mtp.n_max = 4;

    params.stages = { ngram, mtp };

    assert(params.get_max_stage_n_max() == 7);
    assert(params.get_max_verify_batch_tokens() == 8);
    assert(params.get_min_usable_stage_n_min() == 0);
}

int main() {
    test_composite_verify_capacity();
    test_stage_override_controls_capacity();
    std::puts("OK");
    return 0;
}
