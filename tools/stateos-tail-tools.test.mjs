// node --test tools/stateos-tail-tools.test.mjs
// Parser tests for the SV2-E1 census tool and the pure parts of the gate driver (synthetic lines only).

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { pathToFileURL } from "node:url";

import {
  bucketOf,
  checkStep,
  feed,
  newCensus,
  parseFields,
  summarize,
} from "./stateos-div-census.mjs";
import { DEFAULT_LIB, forcedPrompt, score } from "./stateos-tail-gate.mjs";

test("bucketOf matches the C++ buckets", () => {
  assert.deepEqual([1, 2, 5, 6, 64, 65, 512, 513].map(bucketOf), [
    "1",
    "2-5",
    "2-5",
    "6-64",
    "6-64",
    "65-512",
    "65-512",
    ">512",
  ]);
});

test("legacy lines: restores bucketed by tail distance, forced events keep their common prefix", () => {
  const c = newCensus();
  const lines = [
    "prompt eval time =     590.55 ms /    41 tokens (   14.40 ms per token,    69.43 tokens per second)",
    "       eval time =     428.38 ms /    16 tokens (   26.77 ms per token,    37.35 tokens per second)",
    "======== Cache: cache_size = 28845, n_past =  28844, n_past_prompt = 28844",
    "slot apply_checkp: id  0 | task 15405 | restored context checkpoint took  11.87 ms (pos_min = 27643, pos_max = 27643, n_tokens = 27644, n_past = 27644, size = 112.790 MiB)",
    "======== Cache: cache_size = 56, n_past =  41, n_past_prompt = 41",
    "slot apply_checkp: id  0 | task 16 | forcing full prompt re-processing due to lack of cache data (likely due to SWA, see x)",
  ];
  for (const l of lines) feed(c, l);
  const s = summarize(c);
  assert.equal(s.legacy.timedRequests, 1);
  assert.equal(s.legacy.restores, 1);
  assert.equal(s.legacy.byBucket["1"].n, 1);
  assert.equal(s.legacy.byBucket["1"].gapTokens, 28844 - 27644);
  assert.equal(s.legacy.forced, 1);
  assert.equal(s.legacy.forcedPrefixLt64, 1);
  assert.equal(s.v2.divergenceEvents, 0);
});

test("parseFields keeps quoted token windows intact", () => {
  const f = parseFields(
    '[stateos-div] event=restore slot=0 tail_dist=1 cache_win=[151645:"<|im_end|>" *198:"\\n"] prompt_win=[1:"a b"]',
  );
  assert.equal(f.event, "restore");
  assert.equal(f.tail_dist, "1");
  assert.equal(f.cache_win, '[151645:"<|im_end|>" *198:"\\n"]');
  assert.equal(f.prompt_win, '[1:"a b"]');
});

const restore = (fields) =>
  "[stateos-div] event=restore slot=0 task=7 cache_n=2000 n_past=1999 n_past_prompt=1999 tail_dist=1 bucket=1 class=last-token:other" +
  ` prev_stop=eog prev_round=drafted prev_n_draft=4 prev_n_acc=3 prev_cached_after_stop=-1 prev_n_decoded=300 ${fields} cache_win=[] prompt_win=[]`;

test("stateos-div lines: tail availability, hit rate, gaps, crosscheck rows", () => {
  const c = newCensus();
  feed(
    c,
    "[stateos-div] event=create slot=0 task=6 origin=tail pos_min=1995 pos_max=1995 n_tokens=1996 bytes=118 ms=20 n_ckpt=4",
  );
  feed(
    c,
    restore(
      "chosen_origin=tail chosen_pos_max=1995 gap=3 restore_ms=12 reason=tail outcome=restored",
    ),
  );
  feed(
    c,
    "[stateos-div] event=tail_skip slot=0 task=7 cause=no-accepted-draft shadow_pos=10 cache_pos_max=11",
  );
  feed(
    c,
    restore(
      "chosen_origin=tolerance chosen_pos_max=1700 gap=298 restore_ms=12 reason=tolerance outcome=restored",
    ),
  );
  feed(
    c,
    "[stateos-div] event=create slot=0 task=8 origin=tail pos_min=1995 pos_max=1995 n_tokens=1996 bytes=118 ms=20 n_ckpt=4",
  );
  feed(
    c,
    restore(
      "chosen_origin=tail chosen_pos_max=1995 gap=3 restore_ms=12 reason=gen-interval outcome=restored:xcheck-flag-off",
    ),
  );
  feed(
    c,
    "[ckpt-xcheck] origin=tail slot=0 task=8 x=1998 tail_pos=1995 ref_origin=gen-interval ref_pos=1800 layer=3 type=0 n_bitequal=10 n=20 relL2=0.05",
  );
  feed(
    c,
    "[ckpt-xcheck] origin=tail slot=0 task=8 skip=flag-off-reset x=1 tail_pos=0 ref_origin=none ref_pos=-1",
  );
  const s = summarize(c).v2;
  assert.equal(s.divergenceEvents, 3);
  assert.equal(s.lastToken.n, 3);
  assert.equal(s.lastToken.afterDraftedRoundWithAccepted, 3);
  assert.equal(s.lastToken.withTailAvailable, 2);
  assert.equal(s.lastToken.restoredFromTail, 2);
  assert.equal(s.lastToken.tailHitRate, 1);
  assert.equal(s.lastToken.gapTokens, 304);
  assert.deepEqual(s.tailSkips, { "no-accepted-draft": 1 });
  assert.equal(s.creates.tail, 2);
  assert.equal(s.restoreFailedAfterTailChoice, 0);
  assert.equal(s.xcheck.rows, 1);
  assert.equal(s.xcheck.relL2Max, 0.05);
  assert.deepEqual(s.xcheck.skips, { "flag-off-reset": 1 });
});

test("ple-hist lines and the step checks (auto-stop)", () => {
  const armed = [
    "[ple-hist] set seq=0 next_pos=1996 n_prev=2 site=server-resume",
    "[ple-hist] reset seq=0 pos=0 next_pos=-1",
  ];
  const c = newCensus();
  for (const l of armed) feed(c, l);
  feed(
    c,
    "[stateos-div] event=create slot=0 task=6 origin=tail pos_min=1995 pos_max=1995 n_tokens=1996 bytes=1 ms=1 n_ckpt=2",
  );
  feed(
    c,
    restore(
      "chosen_origin=tail chosen_pos_max=1995 gap=3 restore_ms=12 reason=tail outcome=restored",
    ),
  );
  let s = summarize(c);
  assert.equal(s.ple.sets, 1);
  assert.equal(s.ple.resetsAtPos0, 1);
  assert.equal(s.ple.resetsAfterPos0, 0);
  assert.equal(checkStep("step1", s).pass, true);
  assert.equal(checkStep("step2", s).pass, true);

  // an unrepaired rewind stops the chain
  feed(c, "[ple-hist] reset seq=0 pos=1996 next_pos=2001");
  s = summarize(c);
  const r = checkStep("step2", s);
  assert.equal(r.pass, false);
  assert.equal(r.checks.find((x) => x.name.startsWith("ple-hist reset")).ok, false);

  // the repair not armed (no set lines) also stops it, and so does a CUDA error
  const bare = newCensus();
  feed(
    bare,
    restore(
      "chosen_origin=tolerance chosen_pos_max=1700 gap=298 restore_ms=12 reason=tolerance outcome=restored",
    ),
  );
  feed(bare, "CUDA error: an illegal memory access was encountered");
  const b = summarize(bare);
  assert.equal(b.cudaErrors, 1);
  assert.equal(checkStep("step1", b).pass, false);

  // step 3: T1's last-token gap tokens must be >= 90% below P0's
  const p0 = newCensus();
  for (const l of armed) feed(p0, l);
  feed(
    p0,
    restore(
      "chosen_origin=tolerance chosen_pos_max=1700 gap=298 restore_ms=12 reason=tolerance outcome=restored",
    ),
  );
  const t1 = newCensus();
  for (const l of armed) feed(t1, l);
  feed(
    t1,
    restore(
      "chosen_origin=tail chosen_pos_max=1995 gap=3 restore_ms=12 reason=tail outcome=restored",
    ),
  );
  assert.equal(checkStep("step3", summarize(t1), summarize(p0)).pass, true);
  assert.equal(checkStep("step3", summarize(p0), summarize(p0)).pass, false);
});

test("forcedPrompt replaces the last cached generated token", () => {
  const f = forcedPrompt([1, 2, 3], [10, 11, 12, 13], [12, 99]);
  // cached after request A: prompt + g[0..G-2] = [1,2,3,10,11,12]; request B diverges at index 5
  assert.deepEqual(f.tokens, [1, 2, 3, 10, 11, 99]);
  assert.equal(f.forcedIndex, 5);
  assert.equal(f.originalToken, 12);
  assert.equal(forcedPrompt([1], [5], [7]), null);
});

test("score: void on A1 != A0, otherwise the v2 rule", {
  skip: !fs.existsSync(DEFAULT_LIB),
}, async () => {
  const lib = await import(pathToFileURL(DEFAULT_LIB).href);
  const arm = (id, conts) => ({
    arm: id,
    horizon: 8,
    prompts: conts.map((c, i) => ({ id: `p${i}`, ok: true, continuation: c })),
  });
  const base = [
    [1, 2, 3, 4, 5, 6, 7, 8],
    [1, 2, 3, 4, 5, 6, 7, 8],
  ];
  const off = [
    [1, 2, 3, 9, 5, 6, 7, 8],
    [1, 2, 9, 4, 5, 6, 7, 8],
  ];
  const records = [
    arm("A0", base),
    arm("A1", base),
    arm("C", base),
    arm("A5", off),
    arm("A3", off),
  ];
  const opts = { shell: "C", benign: ["A5", "A3"], horizon: 8, minPrompts: 2, nMin: 1 };
  const ok = score(records, lib, opts);
  assert.equal(ok.determinismFailures, 0);
  assert.equal(ok.verdict, "compatible-at-horizon");
  const bad = score(
    [arm("A0", base), arm("A1", off), arm("C", base), arm("A5", off), arm("A3", off)],
    lib,
    opts,
  );
  assert.equal(bad.verdict, "void-determinism");
});
