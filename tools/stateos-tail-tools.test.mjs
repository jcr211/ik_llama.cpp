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
import {
  DEFAULT_LIB,
  engagementAndDrops,
  forcedPrompt,
  score,
  tokensOf,
} from "./stateos-tail-gate.mjs";

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

const ARMED = [
  "[ple-hist] set seq=0 next_pos=1996 n_prev=2 site=server-resume",
  "[ple-hist] reset seq=0 pos=0 next_pos=-1",
];
const TAIL_CREATE =
  "[stateos-div] event=create slot=0 task=6 origin=tail pos_min=1995 pos_max=1995 n_tokens=1996 bytes=1 ms=1 n_ckpt=2";
const TAIL_HIT =
  "chosen_origin=tail chosen_pos_max=1995 gap=3 restore_ms=12 reason=tail outcome=restored";
const FLAG_OFF =
  "chosen_origin=tolerance chosen_pos_max=1700 gap=298 restore_ms=12 reason=tolerance outcome=restored";

const census = (lines) => {
  const c = newCensus();
  for (const l of lines) feed(c, l);
  return summarize(c);
};

test("ple-hist lines, CUDA errors and the step-1/step-2 checks", () => {
  let s = census([...ARMED, TAIL_CREATE, restore(TAIL_HIT)]);
  assert.equal(s.ple.sets, 1);
  assert.equal(s.ple.resetsAtPos0, 1);
  assert.equal(s.ple.resetsAfterPos0, 0);
  assert.equal(checkStep("step1", s).pass, true);
  assert.equal(checkStep("step2", s).pass, true);

  // an unrepaired rewind stops the chain
  s = census([
    ...ARMED,
    TAIL_CREATE,
    restore(TAIL_HIT),
    "[ple-hist] reset seq=0 pos=1996 next_pos=2001",
  ]);
  const r = checkStep("step2", s);
  assert.equal(r.pass, false);
  assert.equal(r.checks.find((x) => x.name.startsWith("ple-hist reset")).ok, false);

  // the repair not armed (no set lines) also stops it, and so does a CUDA error
  const b = census([restore(FLAG_OFF), "CUDA error: an illegal memory access was encountered"]);
  assert.equal(b.cudaErrors, 1);
  assert.equal(checkStep("step1", b).pass, false);
});

test("step 1 excludes new-conversation resets from the denominator", () => {
  const newConv =
    "[stateos-div] event=restore slot=0 task=9 cache_n=500 n_past=41 n_past_prompt=41 tail_dist=459 bucket=>512 class=interior" +
    " prev_stop=eog prev_round=drafted prev_n_draft=4 prev_n_acc=3 prev_cached_after_stop=-1 prev_n_decoded=30" +
    " chosen_origin=none chosen_pos_max=-1 gap=41 restore_ms=0.00 reason=none outcome=reset:no-checkpoint cache_win=[] prompt_win=[]";
  const s = census([...ARMED, restore(FLAG_OFF), newConv, newConv, newConv, newConv, newConv]);
  assert.equal(s.v2.restoreDecisions, 6);
  assert.equal(s.v2.newConversationResets, 5);
  assert.equal(s.v2.divergenceEvents, 1);
  assert.equal(s.v2.lastToken.shareOfEvents, 1);
  assert.equal(checkStep("step1", s).pass, true);
});

test("step 2: tail failures, sha mismatches and writer skips are misses; xcheck-flag-off reset is a hit", () => {
  const failed = census([
    ...ARMED,
    TAIL_CREATE,
    restore(
      "chosen_origin=tail chosen_pos_max=1995 gap=1999 restore_ms=0 reason=none outcome=reset:restore-failed",
    ),
  ]);
  assert.equal(failed.v2.failedAfterTailChoice, 1);
  assert.equal(checkStep("step2", failed).pass, false);

  const sha = census([
    ...ARMED,
    TAIL_CREATE,
    "[stateos-div] event=tail_sha_mismatch slot=0 task=7 pos_max=1995",
    restore(TAIL_HIT),
  ]);
  assert.equal(checkStep("step2", sha).pass, false);

  // a writer refusal counts in the hit-rate denominator: 1 hit of 1 available + 1 refused = 0.5
  const refused = census([
    ...ARMED,
    "[stateos-div] event=tail_skip slot=0 task=5 cause=refused shadow_pos=900 cache_pos_max=905",
    TAIL_CREATE,
    restore(TAIL_HIT),
  ]);
  assert.equal(refused.v2.lastToken.tailHitRate, 0.5);
  assert.equal(checkStep("step2", refused).pass, false);

  const xoff = census([
    ...ARMED,
    TAIL_CREATE,
    restore(
      "chosen_origin=tail chosen_pos_max=1995 gap=1999 restore_ms=5 reason=none outcome=reset:xcheck-flag-off",
    ),
  ]);
  assert.equal(xoff.v2.lastToken.restoredFromTail, 1);
  assert.equal(xoff.v2.failedAfterTailChoice, 0);
  assert.equal(checkStep("step2", xoff).pass, true);

  // a tail_skip after a tail create clears the pending tail: the next divergence is not "available"
  const cleared = census([
    ...ARMED,
    TAIL_CREATE,
    "[stateos-div] event=tail_skip slot=0 task=7 cause=not-newer shadow_pos=10 cache_pos_max=11",
    restore(FLAG_OFF),
  ]);
  assert.equal(cleared.v2.lastToken.withTailAvailable, 0);
});

test("step 3: eligible subclass per event, report-only all-events ratio, no vacuous pass", () => {
  const p0Lines = [...ARMED];
  const t1Lines = [...ARMED];
  const ineligible = (fields) =>
    restore(fields).replace("prev_round=drafted", "prev_round=root-only");
  for (let i = 0; i < 6; i += 1) {
    p0Lines.push(restore(FLAG_OFF)); // eligible in P0 (drafted, n_acc >= 1)
    t1Lines.push(TAIL_CREATE, restore(TAIL_HIT)); // C1's own eligible events
  }
  // ineligible events: large gaps in both runs (no tail possible)
  for (let i = 0; i < 6; i += 1) {
    p0Lines.push(ineligible(FLAG_OFF));
    t1Lines.push(
      "[stateos-div] event=tail_skip slot=0 task=1 cause=no-accepted-draft",
      ineligible(FLAG_OFF),
    );
  }
  const p0 = census(p0Lines);
  const t1 = census(t1Lines);
  assert.equal(p0.v2.lastToken.eligible.n, 6);
  assert.equal(t1.v2.lastToken.eligible.n, 6);
  const r = checkStep("step3", t1, p0);
  assert.equal(r.pass, true, JSON.stringify(r.checks.filter((c) => !c.ok)));
  // the all-events ratio (~0.5 here) would have failed a 90 % test: it is report-only
  const report = r.checks.find((c) => c.reportOnly);
  assert.ok(report && /ratio=0\.5/.test(report.detail));

  // T1 with no restore lines (flag not set / server died) never passes
  const empty = census([...ARMED, TAIL_CREATE]);
  assert.equal(checkStep("step3", empty, p0).pass, false);
  // T1 with far fewer events than P0 never passes
  const starved = census([...ARMED, TAIL_CREATE, restore(TAIL_HIT)]);
  assert.equal(checkStep("step3", starved, p0).pass, false);
  // P0 compared with itself: no reduction, and P0 is not a tail-on run
  assert.equal(checkStep("step3", p0, p0).pass, false);
  // a failed tail restore in T1 is a miss even when the gap numbers pass
  const withFail = census([
    ...t1Lines,
    TAIL_CREATE,
    restore(
      "chosen_origin=tail chosen_pos_max=1995 gap=3 restore_ms=0 reason=none outcome=reset:verify-failed",
    ),
  ]);
  assert.equal(checkStep("step3", withFail, p0).pass, false);
});

// Shape of POST /v1/completions (non-streamed, logprobs: 1), built from this fork's
// server_task_result_cmpl_final::to_json_oaicompat_final + completion_token_output::probs_vector_to_json.
const V1_RESPONSE = {
  choices: [
    {
      text: " Hello",
      index: 0,
      logprobs: {
        content: [
          {
            id: 21806,
            token: " Hel",
            bytes: [32, 72, 101, 108],
            logprob: -0.01,
            top_logprobs: [{ id: 21806, token: " Hel", bytes: [32, 72, 101, 108], logprob: -0.01 }],
          },
          {
            id: 385,
            token: "lo",
            bytes: [108, 111],
            logprob: -0.2,
            top_logprobs: [{ id: 385, token: "lo", bytes: [108, 111], logprob: -0.2 }],
          },
        ],
      },
      finish_reason: "length",
    },
  ],
  created: 1,
  model: "m",
  object: "text_completion",
  usage: { completion_tokens: 2, prompt_tokens: 3, total_tokens: 5 },
  id: "cmpl-x",
};

test("tokensOf reads ids from /v1/completions logprobs and refuses anything else", () => {
  assert.deepEqual(tokensOf(V1_RESPONSE), { ids: [21806, 385], texts: [" Hel", "lo"] });
  // the fork's /completion shape has no ids: refused
  assert.throws(() =>
    tokensOf({ content: "x", completion_probabilities: [{ content: "x", probs: [] }] }),
  );
  const noId = structuredClone(V1_RESPONSE);
  delete noId.choices[0].logprobs.content[1].id;
  assert.throws(() => tokensOf(noId), /integer id/);
  const floatId = structuredClone(V1_RESPONSE);
  floatId.choices[0].logprobs.content[0].id = 1.5;
  assert.throws(() => tokensOf(floatId), /integer id/);
  const noLogprobs = structuredClone(V1_RESPONSE);
  noLogprobs.choices[0].logprobs = null;
  assert.throws(() => tokensOf(noLogprobs), /logprobs/);
  // zero generated tokens: logprobs null and empty text
  assert.deepEqual(tokensOf({ choices: [{ text: "", logprobs: null }] }), { ids: [], texts: [] });
});

test("forcedPrompt replaces the last cached generated token with an unrelated one", () => {
  const cands = [
    { id: 12, text: "\n" },
    { id: 50, text: " th" },
    { id: 99, text: "Z" },
  ];
  // original "\n" (id 12): the same id is refused, " th" is unrelated and chosen
  const f = forcedPrompt([1, 2, 3], [10, 11, 12, 13], cands, "\n");
  assert.deepEqual(f.tokens, [1, 2, 3, 10, 11, 50]);
  // original " the" (id 7): "\n" is unrelated and chosen; " th" (a prefix of " the") would be refused
  const g = forcedPrompt([1, 2, 3], [10, 11, 7, 13], cands, " the");
  assert.equal(forcedPrompt([1, 2, 3], [10, 11, 7, 13], [cands[1]], " the"), null);
  // cached after request A: prompt + g[0..G-2] = [1,2,3,10,11,7]; request B diverges at index 5
  assert.deepEqual(g.tokens, [1, 2, 3, 10, 11, 12]);
  assert.equal(g.forcedIndex, 5);
  assert.equal(g.originalToken, 7);
  assert.equal(forcedPrompt([1], [5], cands, ""), null);
});

test("engagement and per-prompt drops before scoring", () => {
  const rec = (arm) => ({
    arm,
    prompts: [
      { id: "p0", ok: true, forcedIndex: 100 },
      { id: "p1", ok: true, forcedIndex: 200 },
      { id: "p2", ok: true, forcedIndex: 300 },
    ],
  });
  const line = (nPast, tailDist, chosenOrigin, outcome) => ({
    nPast,
    tailDist,
    chosenOrigin,
    outcome,
  });
  const off = [
    line(5, 400, "none", "reset:no-checkpoint"),
    line(100, 1, "tolerance", "restored"),
    line(200, 1, "tolerance", "restored"),
    line(300, 3, "tolerance", "restored"),
  ];
  const on = [
    line(100, 1, "tail", "restored"),
    line(200, 1, "tolerance", "restored"),
    line(300, 1, "tail", "restored"),
  ];
  const records = ["A0", "A1", "C", "A5", "A3"].map(rec);
  const restoresByArm = { A0: off, A1: off, C: on, A5: off, A3: off };
  const r = engagementAndDrops(records, restoresByArm, { shell: "C", minPrompts: 2 });
  assert.equal(r.tailRestores, 2);
  assert.equal(r.engaged, true);
  // p1: C did not restore from the tail; p2: flag-off arms landed at tail_dist 3
  assert.deepEqual([...r.drop.keys()].sort(), ["p1", "p2"]);
  assert.equal(r.droppedPrompts, 2);
  assert.equal(r.droppedCounts["C: not restored from the tail (tolerance/restored)"], 1);
  assert.equal(r.droppedCounts["A0: tail_dist=3"], 1);
  const r2 = engagementAndDrops(records, restoresByArm, { shell: "C", minPrompts: 20 });
  assert.equal(r2.engaged, false);
});

test("score: void on A1 != A0, otherwise the v2 rule; dropped prompts are excluded", {
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
  const withDrop = score(records, lib, { ...opts, drop: new Map([["p1", "C: tail_dist=2"]]) });
  assert.equal(withDrop.rows.length, 1);
  assert.deepEqual(withDrop.dropped, [{ id: "p1", reason: "C: tail_dist=2" }]);
  assert.equal(withDrop.verdict, "insufficient-sample");
});
