# stateos-lane1 — REPORT (from the lane's final message, 2026-09-23)

Base d583c220 (lane 0). Commits 5cfb3c85, a8967890, 2555da57, 8b6b16d2, b401d045, 1fef747d, ce691681, 6330f41c
(+ progress). No GPU used. Build: `.lane/build-l1.cmd` exit 0 (`build-stateos-l1`, Ninja, CUDA arch 120,
AVX512, tests on). Tests: `.lane/test-l1.cmd` exit 0: test-stateos-header 156 checks, 0 failures; ctest 2/2.

Container v1 (one file per save): 'LSOS' | v1 | header_len | header text, then sections TOKS (int32 ids), MAIN
(llama seq state, streamed), CKPT (checkpoints with pos_max < pos_next), optional COMP (companion sub-header + state),
END at EOF. Written to .tmp and renamed. Header lines `H|S|I key=value`; the server's class wins (a file can't demote
a hard field); unknown hard → refuse, unknown soft → warn.
Hard: model_fingerprint (sha256 of the GGUF header + size + 3×64 KiB samples), n_ctx, cache_type_k/v, rope,
kv_layout_version, system_prompt_sha256, kv_geometry, n_tokens + token_sha256 (recomputed from TOKS).
Soft: build. Info: saved_unix, slot_id.
Restore: verify fully (structure, header, TOKS hash, identity, n_tokens ≤ n_ctx, vocab range, CKPT, COMP verify) →
409 with slot_untouched:true and type state_refused (+ refused_field, refused[]) / state_legacy_unkeyed /
state_corrupt / state_missing. Then ONE destructive op: bounded MAIN range-load → MTP invalidate from 0 + warmed-heads
reset (lane 0 only invalidated from pos_next) → clear checkpoints + server_cached_prompt → cache_tokens from TOKS →
COMP load or "skipped: reason" → reinstall checkpoints. A MAIN failure mid-load clears the slot (500,
slot_untouched:false). Success adds a `stateos` object (token_sha256, bytes, companion, warnings, checkpoints_restored).
Also: a pre-existing bug fixed (SLOT_* task errors never reached HTTP → hung requests; now send_slot_error);
`GET /props` `stateos: {version:1, keyed_header:true, companion}`; /list shows State-OS files; new llama APIs
llama_state_seq_append_to_file / load_file_range / layout_desc; common MTP companion accessors.
Not verified (GPU): greedy identity at 4K/32K, live 409s, companion, bytes at 32K/190K, /props on the patched server
→ `.lane/GPU-VERIFY.md` + `.lane/gpu-verify-l1.ps1` (port 8101, auto-stop, standing :8099 relaunched in finally).
Known limits: the new llama APIs are exercised only via the server; report-only non-identical components (kp_l,
sampler, ngram_mod, hidden-state cache → the first post-restore step drafts nothing); corruption inside a structurally
valid MAIN is caught after seq_rm (slot cleared, 500), so a content checksum is v2; save answers 501 for media slots.
