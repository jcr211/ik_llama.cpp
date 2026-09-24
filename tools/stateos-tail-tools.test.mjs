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
  gateStatus,
  preScoreChecks,
  matchRestores,
  score,
  tokensOf,
  tokensSha,
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

const XCHECK_ROW =
  "[ckpt-xcheck] origin=tail slot=0 task=8 x=1998 tail_pos=1995 ref_origin=gen-interval ref_pos=1800 layer=3 type=0 n_bitequal=10 n=20 relL2=0.05";

const census = (lines) => {
  const c = newCensus();
  for (const l of lines) feed(c, l);
  return summarize(c);
};

test("ple-hist lines, CUDA errors and the step-1/step-2 checks", () => {
  // a step-2 probe runs with the crosscheck: its log carries [ckpt-xcheck] rows
  let s = census([...ARMED, TAIL_CREATE, restore(TAIL_HIT), XCHECK_ROW]);
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

test("step 3: one traffic-defined class, T1 mechanism check, VOID on traffic mismatch, no vacuous pass", () => {
  const ineligible = (fields) =>
    restore(fields).replace("prev_round=drafted", "prev_round=root-only");
  const run = (nElig, eligLine, withTail) => {
    const lines = [...ARMED];
    for (let i = 0; i < nElig; i += 1) {
      if (withTail) lines.push(TAIL_CREATE);
      lines.push(restore(eligLine));
    }
    for (let i = 0; i < 6; i += 1) {
      if (withTail)
        lines.push("[stateos-div] event=tail_skip slot=0 task=1 cause=no-accepted-draft");
      lines.push(ineligible(FLAG_OFF));
    }
    return census(lines);
  };
  const p0 = run(6, FLAG_OFF, false);
  const t1 = run(6, TAIL_HIT, true);
  assert.equal(p0.v2.lastToken.eligible.n, 6);
  assert.equal(t1.v2.lastToken.eligible.n, 6);
  assert.equal(p0.v2.lastToken.eligible.definition, t1.v2.lastToken.eligible.definition);
  assert.equal(t1.v2.lastToken.eligible.servedRate, 1);
  const r = checkStep("step3", t1, p0);
  assert.equal(r.verdict, "PASS", JSON.stringify(r.checks.filter((c) => !c.ok)));
  // the all-events ratio (~0.5 here) would have failed a 90 % test: it is report-only
  const report = r.checks.find((c) => c.reportOnly && /all last-token/.test(c.name));
  assert.ok(report && /ratio=0\.5/.test(report.detail));

  // mechanism: eligible events where the tail was not written or not chosen -> STOP
  const lines = [...ARMED];
  for (let i = 0; i < 6; i += 1) lines.push(TAIL_CREATE, restore(TAIL_HIT));
  for (let i = 0; i < 2; i += 1)
    lines.push("[stateos-div] event=tail_skip slot=0 task=1 cause=not-newer", restore(FLAG_OFF));
  const partial = census(lines);
  assert.equal(partial.v2.lastToken.eligible.n, 8);
  assert.equal(partial.v2.lastToken.eligible.servedRate, 0.75);
  assert.equal(checkStep("step3", partial, p0).verdict, "STOP");

  // traffic mismatch -> VOID, not a kill: too few eligible events, or T1's count outside 0.5x-2x
  assert.equal(checkStep("step3", run(3, TAIL_HIT, true), p0).verdict, "VOID");
  assert.equal(checkStep("step3", run(13, TAIL_HIT, true), p0).verdict, "VOID");
  assert.equal(checkStep("step3", run(12, TAIL_HIT, true), p0).verdict, "PASS");
  // T1 with no restore lines never passes (VOID)
  const empty = census([...ARMED, TAIL_CREATE]);
  assert.equal(checkStep("step3", empty, p0).verdict, "VOID");
  // P0 compared with itself: protocol error (T1 tail off) -> VOID
  assert.equal(checkStep("step3", p0, p0).verdict, "VOID");
  // an inert lever (tails written but gaps unchanged) -> STOP
  const inert = run(
    6,
    "chosen_origin=tail chosen_pos_max=1700 gap=298 restore_ms=12 reason=tail outcome=restored",
    true,
  );
  assert.equal(checkStep("step3", inert, p0).verdict, "STOP");
  // a failed tail restore in T1 is a miss
  const withFail = census([
    ...ARMED,
    ...Array.from({ length: 6 }, () => [TAIL_CREATE, restore(TAIL_HIT)]).flat(),
    TAIL_CREATE,
    restore(
      "chosen_origin=tail chosen_pos_max=1995 gap=3 restore_ms=0 reason=none outcome=reset:verify-failed",
    ),
  ]);
  assert.equal(checkStep("step3", withFail, p0).verdict, "STOP");
});

test("census parses the token windows for restore-line binding", () => {
  const s = newCensus();
  feed(
    s,
    '[stateos-div] event=restore slot=0 task=3 cache_n=6 n_past=5 n_past_prompt=5 tail_dist=1 bucket=1 class=last-token:other prev_stop=n_predict prev_round=drafted prev_n_draft=4 prev_n_acc=2 prev_cached_after_stop=-1 prev_n_decoded=64 chosen_origin=tail chosen_pos_max=3 gap=1 restore_ms=10 reason=tail outcome=restored cache_win=[10:"a" 11:"]b" *7:" the"] prompt_win=[10:"a" 11:"]b" *99:"Z"]',
  );
  const r = s.v2.restores[0];
  assert.deepEqual(r.cacheWin, { ids: [10, 11, 7], center: 2 });
  assert.deepEqual(r.promptWin, { ids: [10, 11, 99], center: 2 });
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
  // a UTF-8-split token gets no logprobs entry: 2 ids for 3 completion tokens is refused
  const split = structuredClone(V1_RESPONSE);
  split.usage.completion_tokens = 3;
  assert.throws(() => tokensOf(split), /completion tokens/);
  const noUsage = structuredClone(V1_RESPONSE);
  delete noUsage.usage;
  assert.throws(() => tokensOf(noUsage), /usage/);
  // zero generated tokens: logprobs null, empty text, usage 0
  assert.deepEqual(
    tokensOf({ choices: [{ text: "", logprobs: null }], usage: { completion_tokens: 0 } }),
    {
      ids: [],
      texts: [],
    },
  );
});

test("forcedPrompt replaces the last cached generated token with an unrelated one", () => {
  const cands = [
    { id: 12, text: "\n" },
    { id: 50, text: " th" },
    { id: 99, text: "Z" },
  ];
  // original "\n" (id 12): same id refused; " th" is unrelated (normalized "th" vs "") and chosen
  const f = forcedPrompt([1, 2, 3], [10, 11, 12, 13], cands, "\n");
  assert.deepEqual(f.tokens, [1, 2, 3, 10, 11, 50]);
  // original " the" (id 7): "\n" has no text after normalization, " th" is a prefix: "Z" chosen
  const g = forcedPrompt([1, 2, 3], [10, 11, 7, 13], cands, " the");
  // cached after request A: prompt + g[0..G-2] = [1,2,3,10,11,7]; request B diverges at index 5
  assert.deepEqual(g.tokens, [1, 2, 3, 10, 11, 99]);
  assert.equal(g.forcedIndex, 5);
  assert.equal(g.originalToken, 7);
  // whitespace-insensitive relation: original "\nthe" vs candidate " the" normalize to the same text
  assert.equal(forcedPrompt([1], [10, 7, 13], [{ id: 50, text: " the" }], "\nthe"), null);
  assert.equal(forcedPrompt([1], [5], cands, ""), null);
});

// ---- step 4: one B prompt for every arm, drops, engagement ---------------------------------------

const B = [1, 2, 3, 10, 11, 99];
const row = (id, over = {}) => ({
  id,
  ok: true,
  generated: [10, 11, 7, 13],
  forcedIndex: 5,
  forcedToken: 99,
  originalToken: 7,
  bSha: tokensSha(B),
  continuation: [1, 2, 3, 4, 5, 6, 7, 8],
  ...over,
});
const bLine = (over = {}) => ({
  nPast: 5,
  tailDist: 1,
  chosenOrigin: "tolerance",
  reason: "tolerance",
  outcome: "restored",
  tailAvailable: false,
  cacheWin: { ids: [10, 11, 7], center: 2 },
  promptWin: { ids: [10, 11, 99], center: 2 },
  ...over,
});
const tailLine = (over = {}) =>
  bLine({ chosenOrigin: "tail", reason: "tail", tailAvailable: true, ...over });

test("restore lines bind by request order and content, not position alone", () => {
  const rec = { arm: "A0", prompts: [row("p0"), row("p1", { forcedToken: 50 })] };
  // an equal-position line with other content (another prompt's B) is skipped
  const restores = [
    bLine({ promptWin: { ids: [10, 11, 42], center: 2 } }),
    bLine(),
    bLine({ promptWin: { ids: [10, 11, 50], center: 2 } }),
  ];
  const m = matchRestores(rec, restores);
  assert.equal(m.p0, restores[1]);
  assert.equal(m.p1, restores[2]);
  // order: p1's line before p0's match is not reused
  const out = [bLine({ promptWin: { ids: [10, 11, 50], center: 2 } }), bLine()];
  const m2 = matchRestores(rec, out);
  assert.equal(m2.p0, out[1]);
  assert.equal(m2.p1, null);
});

test("engagement and drops: B built once from A0, constraints on A0/A1/C only", () => {
  const ids = ["p0", "p1", "p2", "p3"];
  const mk = (arm, over = {}) => ({ arm, prompts: ids.map((id) => row(id, over[id] ?? {})) });
  const records = [
    mk("A0"),
    mk("A1"),
    // C's request A differed on p2 (its cache is not A0's)
    mk("C", { p2: { generated: [10, 11, 8, 13] } }),
    // a benign arm whose own request A differs everywhere still answers A0's B: not dropped for that
    mk("A5", { p0: { generated: [9, 9, 9, 9] }, p1: { generated: [9, 9, 9, 9] } }),
    // a benign arm that sent a different B (it built its own) is dropped
    mk("A3", { p3: { bSha: tokensSha([1, 2, 3, 9, 9, 99]) } }),
  ];
  const lines = ids.map(() => bLine());
  const restoresByArm = {
    A0: lines,
    A1: lines,
    C: ids.map(() => tailLine()),
    // benign logs carry no B restore lines at tail distance 1: unconstrained
    A5: [],
    A3: [],
  };
  const r = engagementAndDrops(records, restoresByArm, { shell: "C", minPrompts: 2 });
  assert.equal(r.tailRestores, 4);
  assert.equal(r.tailAvailableAtB, 4);
  assert.equal(r.engaged, true);
  assert.deepEqual([...r.drop.keys()].sort(), ["p2", "p3"]);
  assert.equal(r.droppedCounts["C: request A output differs from A0's"], 1);
  assert.equal(r.droppedCounts["A3: request B differs from A0's"], 1);
  // no eligible prompts vs tails not used
  const noTails = engagementAndDrops(
    records,
    { ...restoresByArm, C: ids.map(() => bLine()) },
    { shell: "C", minPrompts: 2 },
  );
  assert.equal(noTails.engaged, false);
  assert.equal(noTails.eligibleExisted, false);
  const unused = engagementAndDrops(
    records,
    { ...restoresByArm, C: ids.map(() => bLine({ tailAvailable: true })) },
    { shell: "C", minPrompts: 2 },
  );
  assert.equal(unused.engaged, false);
  assert.equal(unused.eligibleExisted, true);
  assert.equal(gateStatus("not-engaged:no-eligible-prompts"), "VOID");
  assert.equal(gateStatus("not-engaged:tails-not-used"), "STOP");
  assert.equal(gateStatus("insufficient-sample"), "VOID");
  assert.equal(gateStatus("void-determinism"), "VOID");
  assert.equal(gateStatus("shellWorse"), "STOP");
  assert.equal(gateStatus("compatible-at-horizon"), "PASS");
});

test("score: void on A1 != A0, otherwise the v2 rule; drops and differing B prompts are excluded", {
  skip: !fs.existsSync(DEFAULT_LIB),
}, async () => {
  const lib = await import(pathToFileURL(DEFAULT_LIB).href);
  const arm = (id, conts, bShas = []) => ({
    arm: id,
    horizon: 8,
    prompts: conts.map((c, i) => ({
      id: `p${i}`,
      ok: true,
      continuation: c,
      bSha: bShas[i] ?? "b",
    })),
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
  assert.equal(gateStatus(withDrop.verdict), "VOID");
  // a benign arm that answered a different B on p0 drops p0
  const diffB = score(
    [arm("A0", base), arm("A1", base), arm("C", base), arm("A5", off, ["x"]), arm("A3", off)],
    lib,
    opts,
  );
  assert.deepEqual(diffB.dropped, [{ id: "p0", reason: "request B differs across arms" }]);
});
// ---- round 4 ------------------------------------------------------------------------------------

test("step 3 floor rule: >= 90 % of the ACHIEVABLE reduction, VOID when the gap is too small", () => {
  const withGap = (gap) =>
    `chosen_origin=tolerance chosen_pos_max=1700 gap=${gap} restore_ms=12 reason=tolerance outcome=restored`;
  const p0Of = (gap) =>
    census([...ARMED, ...Array.from({ length: 6 }, () => restore(withGap(gap)))]);
  const t1 = census([
    ...ARMED,
    ...Array.from({ length: 6 }, () => [TAIL_CREATE, restore(TAIL_HIT)]).flat(),
  ]);
  assert.equal(t1.v2.lastToken.eligible.floor, 3);
  // short generations (P0 gap 29, n_acc 3): the raw ratio 3/29 = 10.3 % would have failed; the floor
  // rule passes (bound = 3 + 0.1 x (29 - 3) = 5.6)
  const r = checkStep("step3", t1, p0Of(29));
  assert.equal(r.verdict, "PASS", JSON.stringify(r.checks.filter((c) => !c.ok)));
  const raw = r.checks.find((c) => c.reportOnly && /raw per-event/.test(c.name));
  assert.ok(raw && /T1\/P0=0\.10/.test(raw.detail));
  // nothing meaningful to save: P0 gap 4 <= 1.5 x 3 -> VOID
  assert.equal(checkStep("step3", t1, p0Of(4)).verdict, "VOID");
  // a T1 that pays more than the bound -> STOP
  const slow = census([
    ...ARMED,
    ...Array.from({ length: 6 }, () => [
      TAIL_CREATE,
      restore(
        "chosen_origin=tail chosen_pos_max=1990 gap=9 restore_ms=12 reason=tail outcome=restored",
      ),
    ]).flat(),
  ]);
  assert.equal(checkStep("step3", slow, p0Of(29)).verdict, "STOP");
});

test("mislaunched runs are VOID; a CUDA error is STOP even when the step would be VOID", () => {
  const p0 = census([...ARMED, ...Array.from({ length: 6 }, () => restore(FLAG_OFF))]);
  // T1 served by the crosscheck launcher: the server continued on the flag-off state
  const xT1 = census([
    ...ARMED,
    ...Array.from({ length: 6 }, () => [
      TAIL_CREATE,
      restore(
        "chosen_origin=tail chosen_pos_max=1995 gap=298 restore_ms=12 reason=tolerance outcome=restored:xcheck-flag-off",
      ),
    ]).flat(),
  ]);
  assert.equal(xT1.v2.xcheckActive, true);
  const r = checkStep("step3", xT1, p0);
  assert.equal(r.verdict, "VOID");
  assert.ok(r.checks.find((c) => /crosscheck/.test(c.name) && !c.ok));
  // a step-2 probe launched without -Tail
  const noTail = census([...ARMED, ...Array.from({ length: 6 }, () => restore(FLAG_OFF))]);
  assert.equal(checkStep("step2", noTail).verdict, "VOID");
  // a step-2 probe launched with the tail but without the crosscheck (TAIL_XCHECK unset)
  const tailOnly = census([...ARMED, TAIL_CREATE, restore(TAIL_HIT)]);
  const r2 = checkStep("step2", tailOnly);
  assert.equal(r2.verdict, "VOID");
  assert.ok(r2.checks.find((c) => /crosscheck on/.test(c.name) && !c.ok));
  // the same probe with the crosscheck on passes
  assert.equal(
    checkStep("step2", census([...ARMED, TAIL_CREATE, restore(TAIL_HIT), XCHECK_ROW])).verdict,
    "PASS",
  );
  // CUDA error wins over VOID
  const crashed = census([
    ...ARMED,
    TAIL_CREATE,
    "CUDA error: an illegal memory access was encountered",
  ]);
  assert.equal(checkStep("step3", crashed, p0).verdict, "STOP");
  assert.equal(checkStep("step2", census([...ARMED, "CUDA error: x"])).verdict, "STOP");
});

test("Opus repro: a C arm served by the crosscheck never scores (restored:xcheck-flag-off is no tail restore)", () => {
  const ids = Array.from({ length: 24 }, (_, i) => `p${i}`);
  const mk = (arm) => ({ arm, prompts: ids.map((id) => row(id)) });
  const records = ["A0", "A1", "C", "A5", "A3"].map(mk);
  const xLine = () =>
    bLine({
      chosenOrigin: "tail",
      tailAvailable: true,
      outcome: "restored:xcheck-flag-off",
      reason: "gen-interval",
    });
  const restoresByArm = {
    A0: ids.map(() => bLine()),
    A1: ids.map(() => bLine()),
    C: ids.map(xLine),
    A5: [],
    A3: [],
  };
  const pre = engagementAndDrops(records, restoresByArm, { shell: "C", minPrompts: 20 });
  assert.equal(pre.tailRestores, 0);
  assert.equal(pre.engaged, false);
  // and runScore refuses the arm outright from its log
  const cCensus = newCensus();
  feed(
    cCensus,
    "[ckpt-xcheck] origin=tail slot=0 task=8 skip=flag-off-reset x=1 tail_pos=0 ref_origin=none ref_pos=-1",
  );
  const refused = preScoreChecks(records, { A0: newCensus(), C: cCensus }, { shell: "C" });
  assert.equal(refused.verdict, "mislaunched:xcheck");
  assert.equal(gateStatus(refused.verdict), "VOID");
  const cOutcome = newCensus();
  feed(
    cOutcome,
    restore(
      "chosen_origin=tail chosen_pos_max=1995 gap=298 restore_ms=12 reason=tolerance outcome=restored:xcheck-flag-off",
    ),
  );
  assert.equal(
    preScoreChecks(records, { C: cOutcome }, { shell: "C" }).verdict,
    "mislaunched:xcheck",
  );
});

test("pre-score: CUDA errors in any arm and C-only failures are STOP", () => {
  const ids = ["p0", "p1"];
  const mk = (arm, over = {}) => ({ arm, prompts: ids.map((id) => row(id, over[id] ?? {})) });
  const records = [mk("A0"), mk("A1"), mk("C")];
  const bad = newCensus();
  feed(bad, "CUDA error: an illegal memory access was encountered");
  const r = preScoreChecks(records, { A0: newCensus(), A5: bad, C: newCensus() }, { shell: "C" });
  assert.equal(r.verdict, "cuda-errors");
  assert.equal(gateStatus(r.verdict), "STOP");
  const cFail = [mk("A0"), mk("A1"), mk("C", { p1: { ok: false, error: "HTTP 500" } })];
  const r2 = preScoreChecks(cFail, { A0: newCensus(), C: newCensus() }, { shell: "C" });
  assert.equal(r2.verdict, "shell-errors");
  assert.equal(gateStatus(r2.verdict), "STOP");
  assert.equal(preScoreChecks(records, { A0: newCensus(), C: newCensus() }, { shell: "C" }), null);
});

test("A1's request A differing from A0's is void-determinism, not a drop", {
  skip: !fs.existsSync(DEFAULT_LIB),
}, async () => {
  const lib = await import(pathToFileURL(DEFAULT_LIB).href);
  const ids = ["p0", "p1"];
  const mk = (arm, over = {}) => ({
    arm,
    horizon: 8,
    prompts: ids.map((id) => row(id, over[id] ?? {})),
  });
  const records = [
    mk("A0"),
    mk("A1", { p1: { generated: [10, 11, 8, 13] } }),
    mk("C"),
    mk("A5", {
      p0: { continuation: [1, 2, 3, 9, 5, 6, 7, 8] },
      p1: { continuation: [1, 2, 9, 4, 5, 6, 7, 8] },
    }),
    mk("A3", {
      p0: { continuation: [1, 2, 3, 9, 5, 6, 7, 8] },
      p1: { continuation: [1, 2, 9, 4, 5, 6, 7, 8] },
    }),
  ];
  const lines = ids.map(() => bLine());
  const pre = engagementAndDrops(
    records,
    { A0: lines, A1: [lines[0]], C: ids.map(() => tailLine()), A5: [], A3: [] },
    { shell: "C", minPrompts: 1 },
  );
  assert.equal(pre.drop.has("p1"), false);
  const s = score(records, lib, {
    shell: "C",
    benign: ["A5", "A3"],
    horizon: 8,
    minPrompts: 1,
    nMin: 1,
    drop: pre.drop,
  });
  assert.equal(s.determinismFailures, 1);
  assert.equal(s.verdict, "void-determinism");
  assert.equal(gateStatus(s.verdict), "VOID");
});
