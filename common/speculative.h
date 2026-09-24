#pragma once

#include "llama.h"
#include "llama-spec-features.h"
#include "common.h"
#include "spec-tuner.h"

struct common_speculative;

bool common_speculative_needs_checkpoint(const llama_model * model);

enum common_speculative_init_status {
    COMMON_SPECULATIVE_INIT_SKIPPED,
    COMMON_SPECULATIVE_INIT_READY,
    COMMON_SPECULATIVE_INIT_ERR_RECURRENT,
    COMMON_SPECULATIVE_INIT_ERR_MTP,
    COMMON_SPECULATIVE_INIT_ERR_GENERIC,
};

using common_speculative_feature_kind = llama_spec_feature_kind;
using common_speculative_feature_row_view = llama_spec_feature_row_view;
using common_speculative_feature_view = llama_spec_feature_view;

static constexpr common_speculative_feature_kind COMMON_SPECULATIVE_FEATURE_NONE = LLAMA_SPEC_FEATURE_NONE;
static constexpr common_speculative_feature_kind COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE = LLAMA_SPEC_FEATURE_HIDDEN_STATE;

struct common_speculative_token_dist {
    llama_tokens ids;
    std::vector<float> probs;
};

struct common_speculative_checkpoint {
    bool valid = false;
    int mode = LLAMA_SPEC_CKPT_NONE;
    llama_pos n_past = 0;
    // LONGSPEAR_PLE_HIST_REWIND: the target's PLE n-gram history when the checkpoint was saved
    std::vector<llama_token> ple_hist;
    llama_pos ple_next_pos = -1;
    llama_token sampled = LLAMA_TOKEN_NULL;
    common_sampler * sampler = nullptr;
    bool sampler_borrowed = false; // LONGSPEAR_SPEC_CKPT_LEAN: sampler is the slot's persistent one

    void clear();
};

// LONGSPEAR_SPEC_HOST_TIMING=1 (off by default): host microseconds of one verify round, printed as a
// single [spec-host] line on stderr. restore_result is -1 when the round restored nothing (every
// draft accepted); mtp_skip counts MTP draft skips since the previous verify round. K is the verified
// batch, k_prop = K + clamp the batch the drafter proposed before the checkpoint-capacity clamp.
struct common_speculative_host_timing {
    int     mode             = LLAMA_SPEC_CKPT_NONE;
    int     restore_result   = -1;
    int     redecode_n       = 0;
    int     mtp_skip         = 0;
    int     clamp            = 0;
    int     xcheck           = 0; // LONGSPEAR_SPEC_CKPT_CROSSCHECK redid this round the gpu-fallback way
    int64_t ckpt_init_us     = 0;
    int64_t ckpt_save_us     = 0;
    int64_t save_cells_us    = 0;
    int64_t save_shadow_us   = 0;
    int64_t save_sync_us     = 0;
    int64_t sampler_init_us  = 0;
    int64_t sampler_clone_us = 0;
    int64_t restore_us       = 0;
    int64_t redecode_us      = 0;
    int64_t draft_host_us    = 0;
    int64_t sample_us        = 0;
};

bool common_speculative_host_timing_enabled();

// this slot's record for the round in progress (nullptr without a speculative state)
common_speculative_host_timing * common_speculative_host_timing_get(common_speculative * spec);

// print the [spec-host] line for a finished verify round of n_verify tokens and clear the record
void common_speculative_host_timing_emit(common_speculative * spec, int id_slot, int n_verify, int n_accepted);

// LONGSPEAR_SPEC_CKPT_MAX_TOKENS=M (off when unset or < 2): the per-step checkpoint capacity fixed at
// startup, instead of the longest verify batch the stage chain can build
int common_speculative_ckpt_max_tokens_override();

// LONGSPEAR_SPEC_CLAMP_TO_CKPT=1 with a capacity override M: a draft longer than M-1 tokens keeps its
// first M-1 instead of falling back to a root-only batch that drops every draft. Returns how many
// draft tokens to drop (0 when off or when the draft fits) and counts the clamp.
int common_speculative_ckpt_clamp(common_speculative * spec, const llama_model * model, size_t n_draft);

// the last draft (n_draft tokens) is verified only up to its first n_keep tokens: the drafting stage's
// acceptance bookkeeping (ngram-mod / suffix low-acceptance streaks) and the draft statistics are
// corrected to n_keep. Called by the clamp; public for tests.
void common_speculative_truncate_draft(common_speculative * spec, size_t n_draft, size_t n_keep);

struct common_speculative_draft_result {
    llama_tokens tokens;
    std::vector<common_speculative_token_dist> proposal_dists; // Sparse proposal distributions populated by stochastic DFlash2
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;
    bool target_only = false;
};

struct common_speculative_metrics_stage_snapshot {
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;

    uint64_t n_call_begin = 0;
    uint64_t n_call_draft = 0;
    uint64_t n_call_accept = 0;

    uint64_t n_gen_drafts = 0;
    uint64_t n_acc_drafts = 0;
    uint64_t n_gen_tokens = 0;
    uint64_t n_acc_tokens = 0;

    // Position zero represents speculative position 1.
    std::vector<uint64_t> drafted_by_position;
    std::vector<uint64_t> accepted_by_position;

    int64_t t_begin_us = 0;
    int64_t t_draft_us = 0;
    int64_t t_accept_us = 0;
};

struct common_speculative_metrics_snapshot {
    std::vector<common_speculative_metrics_stage_snapshot> stages;
};

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

common_speculative_init_status common_speculative_try_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        common_speculative      ** out_spec);

void common_speculative_prepare_startup(
        gpt_params & params_base,
        bool         allow_parallel_mtp = true);

bool common_speculative_finalize_startup(
        gpt_params        & params_base,
        const llama_model * model);

bool common_speculative_load_draft_model(
        common_params_speculative & params,
        const gpt_params         & params_base);

bool common_speculative_prepare_mtp_runtime(
        common_params_speculative & params,
        const gpt_params         & params_base,
        const llama_model        * model,
        bool                       has_external_mtp);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// apply per-request runtime parameters before prompt warmup can touch companion state
void common_speculative_prepare_request(common_speculative * spec, common_params_speculative & params);

// true when the active request drafts with more MTP heads than the cached prefix was
// warmed with; the caller must then reprocess the prompt from position 0
bool common_speculative_mtp_requires_fresh_warmup(const common_speculative * spec);

// sample up to n_draft tokens and add them to the batch using the draft model
// draft_base_pos/draft_seq_id override the MTP position for id_last
llama_tokens common_speculative_draft(
                     common_speculative * spec,
                     common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last,
                            llama_pos     draft_base_pos = -1,
                            llama_seq_id  draft_seq_id = 0);

common_speculative_draft_result common_speculative_draft_ex(
                     common_speculative * spec,
                     llama_context * ctx,
                     common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last,
                            llama_pos     draft_base_pos = -1,
                            llama_seq_id  draft_seq_id = 0,
                            const common_params_sampling * sampling = nullptr);

int common_speculative_get_configured_n_max(const common_speculative * spec);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

bool common_speculative_before_draft(
    common_speculative * spec,
    llama_model * model,
    llama_context * ctx,
    common_sampler * sampler_src,
    const common_params_sampling & sparams,
    llama_seq_id seq_id,
    llama_pos n_past,
    llama_token sampled,
    int max_tokens,
    int ckpt_mode);

bool common_speculative_ensure_sequence_hidden(
    common_speculative * spec,
    llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos pos);

bool common_speculative_capture_output_hidden(
    common_speculative * spec,
    llama_context * ctx,
    int32_t output_index,
    llama_seq_id seq_id,
    llama_pos pos);

bool common_speculative_copy_output_hidden_rows(
    const common_speculative * spec,
    llama_context * ctx,
    const std::vector<int32_t> & output_indices,
    std::vector<float> & hidden_rows);

bool common_speculative_commit_accepted_hidden_rows(
    common_speculative * spec,
    common_speculative_type spec_type_used,
    llama_seq_id seq_id,
    llama_pos pos_base,
    llama_token sampled_before,
    const std::vector<llama_token> & ids,
    const std::vector<float> & hidden_rows);

bool common_speculative_commit_accepted_output(
    common_speculative * spec,
    llama_context * ctx,
    common_speculative_type spec_type_used,
    llama_seq_id seq_id,
    llama_pos pos_base,
    llama_token sampled_before,
    const std::vector<llama_token> & ids,
    const std::vector<int32_t> & output_indices);

const common_speculative_checkpoint * common_speculative_get_checkpoint(const common_speculative * spec);

void common_speculative_checkpoint_discard(
    common_speculative_checkpoint & ckpt,
    llama_context * ctx);

bool common_speculative_checkpoint_restore(
    common_speculative_checkpoint & ckpt,
    common_speculative * spec,
    llama_context * ctx,
    common_sampler * sampler_dst,
    llama_seq_id seq_id,
    common_speculative_type spec_type_used,
    llama_token sampled_before,
    const std::vector<llama_token> & ids,
    int n_draft,
    const std::vector<float> & mtp_hidden_state_pre,
    int32_t mtp_n_past_base);

bool common_speculative_commit(
        common_speculative * spec,
        llama_context * ctx,
        common_sampler * sampler_dst,
        llama_seq_id seq_id,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        int n_draft,
        llama_pos pos_base,
        const std::vector<int32_t> & accepted_output_indices);

bool common_speculative_has_sequence_hidden(const common_speculative * spec, llama_seq_id seq_id);

void common_speculative_clear_sequence_hidden(common_speculative * spec, llama_seq_id seq_id);

void common_speculative_clear_sequence(
    common_speculative * spec,
    llama_seq_id seq_id,
    bool clear_companion_ctx = false);

// Invalidate everything the MTP/draft companion holds at or after pos_begin for seq_id: the
// companion KV tail, the cached draft token/embedding and the cached target hidden state.
// Call this after any non-monotonic move of the target context (cache trim, context-checkpoint
// restore) that does not itself rewind the companion. Cheap, and a no-op when the companion is
// already consistent (no extra decode).
void common_speculative_mtp_invalidate(
    common_speculative * spec,
    llama_seq_id seq_id,
    llama_pos pos_begin);

bool common_speculative_trim_sequence(
    common_speculative * spec,
    llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos pos_begin);

void common_speculative_clear_sequence_kv(
    common_speculative * spec,
    llama_context * ctx,
    llama_seq_id seq_id);

llama_context * common_speculative_get_companion_ctx(common_speculative * spec);

int32_t common_speculative_on_target_seq_batch(
    common_speculative * spec,
    llama_context * ctx,
    const llama_batch & batch,
    llama_seq_id seq_id,
    bool is_prompt_warmup);

int32_t common_speculative_on_target_batch(
    common_speculative * spec,
    const llama_batch & batch,
    const common_speculative_feature_view & features,
    bool is_prompt_warmup);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec, double slot_tps = 0.0, int n_decoded = 0, int n_past = 0, common_params_speculative * active_params = nullptr);

common_speculative_type common_speculative_current_type(const common_speculative * spec);

common_speculative_metrics_snapshot common_speculative_get_metrics_snapshot(const common_speculative * spec);

// Context shift for MTP to match how server handle main model
void common_speculative_context_shift(
        common_speculative * spec,
        llama_seq_id         seq_id,
        llama_pos            kv_keep,
        llama_pos            kv_discard,
        llama_pos            kv_past);

struct common_speculative_round_result {
    bool attempted = false;
    bool sampled_before_ready = false;
    bool sampled_before_from_carry = false;
    bool used_speculative = false;
    bool failed = false;
    std::string error;
    llama_token sampled_before = LLAMA_TOKEN_NULL;
    llama_tokens ids;
};

common_speculative_round_result common_speculative_run_round(
    common_speculative * spec,
    llama_model * model,
    llama_context * ctx,
    common_sampler * sampler,
    llama_context * ctx_guidance,
    common_params_speculative params,
    const common_params_sampling & sparams,
    llama_seq_id seq_id,
    llama_pos n_past,
    int n_predict_budget,
    bool have_carry,
    const llama_tokens & draft_history,
    llama_token carry_token);
