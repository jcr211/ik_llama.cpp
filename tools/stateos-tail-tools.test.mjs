// node --test tools/stateos-tail-tools.test.mjs
// Parser tests for the SV2-E1 census tool and the pure parts of the gate driver (synthetic lines only).

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

import {
  bucketOf,
  checkStep,
  feed,
  feedFiles,
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
  preVerdict,
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

// the server's startup line (normally from <LogStem>.out.log) plus an armed PLE-history repair
const LISTENING =
  'INFO [                    main] HTTP server listening | tid="1" timestamp=1 hostname="0.0.0.0" port="8099"';
const ARMED = [
  LISTENING,
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

// the launcher's recorded flags ([stateos-flags], normally from the <LogStem>.flags sidecar)
const flags = (div, tail, xcheck, extra = "") =>
  `[stateos-flags] LONGSPEAR_STATEOS_DIV_LOG=${div} LONGSPEAR_STATEOS_TAIL_SNAPSHOT=${tail} LONGSPEAR_STATEOS_TAIL_XCHECK=${xcheck} LONGSPEAR_PLE_HIST_REWIND=1 LONGSPEAR_PLE_HIST_LOG=1 logstem=x extra='${extra}'`;
const F_OFF = flags(1, 0, 0); // step 1, P0, step-4 A0/A1
const F_PROBE = flags(1, 1, 1); // step 2
const F_TAIL = flags(1, 1, 0); // T1, step-4 C
const F_A5 = flags(1, 0, 0, "-no-fmoe -no-fug"); // step-4 benign A5
const F_A3 = flags(1, 0, 0, "-fa 0"); // step-4 benign A3

test("ple-hist lines, CUDA errors and the step-1/step-2 checks", () => {
  let s = census([F_OFF, ...ARMED, TAIL_CREATE, restore(TAIL_HIT)]);
  assert.equal(s.ple.sets, 1);
  assert.equal(s.ple.resetsAtPos0, 1);
  assert.equal(s.ple.resetsAfterPos0, 0);
  assert.equal(checkStep("step1", s).verdict, "PASS");
  // a step-2 probe runs with the tail and the crosscheck
  s = census([F_PROBE, ...ARMED, TAIL_CREATE, restore(TAIL_HIT), XCHECK_ROW]);
  assert.equal(checkStep("step2", s).verdict, "PASS");

  // an unrepaired rewind stops the chain
  s = census([
    F_PROBE,
    ...ARMED,
    TAIL_CREATE,
    restore(TAIL_HIT),
    "[ple-hist] reset seq=0 pos=1996 next_pos=2001",
  ]);
  const r = checkStep("step2", s);
  assert.equal(r.verdict, "STOP");
  assert.equal(r.checks.find((x) => x.name.startsWith("ple-hist reset")).ok, false);

  // the repair not armed (no set lines) also stops it, and so does a CUDA error
  const b = census([
    F_OFF,
    restore(FLAG_OFF),
    "CUDA error: an illegal memory access was encountered",
  ]);
  assert.equal(b.cudaErrors, 1);
  assert.equal(checkStep("step1", b).verdict, "STOP");
});

test("step 1 excludes new-conversation resets from the denominator", () => {
  const newConv =
    "[stateos-div] event=restore slot=0 task=9 cache_n=500 n_past=41 n_past_prompt=41 tail_dist=459 bucket=>512 class=interior" +
    " prev_stop=eog prev_round=drafted prev_n_draft=4 prev_n_acc=3 prev_cached_after_stop=-1 prev_n_decoded=30" +
    " chosen_origin=none chosen_pos_max=-1 gap=41 restore_ms=0.00 reason=none outcome=reset:no-checkpoint cache_win=[] prompt_win=[]";
  const s = census([
    F_OFF,
    ...ARMED,
    restore(FLAG_OFF),
    newConv,
    newConv,
    newConv,
    newConv,
    newConv,
  ]);
  assert.equal(s.v2.restoreDecisions, 6);
  assert.equal(s.v2.newConversationResets, 5);
  assert.equal(s.v2.divergenceEvents, 1);
  assert.equal(s.v2.lastToken.shareOfEvents, 1);
  assert.equal(checkStep("step1", s).verdict, "PASS");
});

test("step 2: with the right flags every tail failure is STOP; xcheck-flag-off reset is a hit", () => {
  const n30 = (lines) => Array.from({ length: 30 }, () => lines).flat();
  // (1) every tail restore fails verification
  const verifyFail = census([
    F_PROBE,
    ...ARMED,
    ...n30([
      TAIL_CREATE,
      restore(
        "chosen_origin=tail chosen_pos_max=1995 gap=1999 restore_ms=0 reason=none outcome=reset:verify-failed",
      ),
    ]),
  ]);
  assert.equal(verifyFail.v2.failedAfterTailChoice, 30);
  assert.equal(checkStep("step2", verifyFail).verdict, "STOP");
  // (2) tails written but the search never chooses them
  const neverChosen = census([F_PROBE, ...ARMED, ...n30([TAIL_CREATE, restore(FLAG_OFF)])]);
  assert.equal(checkStep("step2", neverChosen).verdict, "STOP");
  // (3) every tail has a sha mismatch
  const shaAll = census([
    F_PROBE,
    ...ARMED,
    ...n30([
      TAIL_CREATE,
      "[stateos-div] event=tail_sha_mismatch slot=0 task=7 pos_max=1995",
      restore(FLAG_OFF),
    ]),
  ]);
  assert.equal(checkStep("step2", shaAll).verdict, "STOP");
  // (4) the writer refuses every tail
  const refusedAll = census([
    F_PROBE,
    ...ARMED,
    ...n30([
      "[stateos-div] event=tail_skip slot=0 task=5 cause=refused shadow_pos=900 cache_pos_max=905",
      restore(FLAG_OFF),
    ]),
  ]);
  assert.equal(checkStep("step2", refusedAll).verdict, "STOP");

  // a writer refusal counts in the hit-rate denominator: 1 hit of 1 available + 1 refused = 0.5
  const refused = census([
    F_PROBE,
    ...ARMED,
    "[stateos-div] event=tail_skip slot=0 task=5 cause=refused shadow_pos=900 cache_pos_max=905",
    TAIL_CREATE,
    restore(TAIL_HIT),
    XCHECK_ROW,
  ]);
  assert.equal(refused.v2.lastToken.tailHitRate, 0.5);
  assert.equal(checkStep("step2", refused).verdict, "STOP");

  const xoff = census([
    F_PROBE,
    ...ARMED,
    TAIL_CREATE,
    restore(
      "chosen_origin=tail chosen_pos_max=1995 gap=1999 restore_ms=5 reason=none outcome=reset:xcheck-flag-off",
    ),
  ]);
  assert.equal(xoff.v2.lastToken.restoredFromTail, 1);
  assert.equal(xoff.v2.failedAfterTailChoice, 0);
  assert.equal(checkStep("step2", xoff).verdict, "PASS");

  // a tail_skip after a tail create clears the pending tail: the next divergence is not "available"
  const cleared = census([
    F_PROBE,
    ...ARMED,
    TAIL_CREATE,
    "[stateos-div] event=tail_skip slot=0 task=7 cause=not-newer shadow_pos=10 cache_pos_max=11",
    restore(FLAG_OFF),
  ]);
  assert.equal(cleared.v2.lastToken.withTailAvailable, 0);
});

test("mislaunch is decided from the recorded flags only: wrong flags or no record = VOID", () => {
  const good = [...ARMED, TAIL_CREATE, restore(TAIL_HIT), XCHECK_ROW];
  // wrong flags: a step-2 probe launched without the crosscheck, or without the tail
  assert.equal(checkStep("step2", census([F_TAIL, ...good])).verdict, "VOID");
  assert.equal(checkStep("step2", census([F_OFF, ...good])).verdict, "VOID");
  // flags unknown (no sidecar, no header line)
  const unknown = checkStep("step2", census(good));
  assert.equal(unknown.verdict, "VOID");
  assert.ok(unknown.checks.find((c) => /flags unknown/.test(c.detail)));
  // step 1 run with the tail on is mislaunched
  assert.equal(checkStep("step1", census([F_TAIL, ...ARMED, restore(FLAG_OFF)])).verdict, "VOID");
  // the probe's own output never decides: right flags, no crosscheck line at all, lever failing -> STOP
  const failing = census([F_PROBE, ...ARMED, TAIL_CREATE, restore(FLAG_OFF)]);
  assert.equal(checkStep("step2", failing).verdict, "STOP");
});

test("step 3: one traffic-defined class, T1 mechanism check, VOID on traffic mismatch, no vacuous pass", () => {
  const ineligible = (fields) =>
    restore(fields).replace("prev_round=drafted", "prev_round=root-only");
  const run = (nElig, eligLine, withTail) => {
    const lines = [withTail ? F_TAIL : F_OFF, ...ARMED];
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
  const lines = [F_TAIL, ...ARMED];
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
  const empty = census([F_TAIL, ...ARMED, TAIL_CREATE]);
  assert.equal(checkStep("step3", empty, p0).verdict, "VOID");
  // P0 compared with itself: T1's recorded flags are the tail-off set -> mislaunched, VOID
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
    F_TAIL,
    ...ARMED,
    ...Array.from({ length: 6 }, () => [TAIL_CREATE, restore(TAIL_HIT)]).flat(),
    TAIL_CREATE,
    restore(
      "chosen_origin=tail chosen_pos_max=1995 gap=3 restore_ms=0 reason=none outcome=reset:verify-failed",
    ),
  ]);
  assert.equal(checkStep("step3", withFail, p0).verdict, "STOP");
  // tail-integrity misses are hard: they STOP even when a traffic check VOIDs the step
  const fewAndBroken = census([
    F_TAIL,
    ...ARMED,
    TAIL_CREATE,
    restore(
      "chosen_origin=tail chosen_pos_max=1995 gap=3 restore_ms=0 reason=none outcome=reset:verify-failed",
    ),
    "[stateos-div] event=tail_sha_mismatch slot=0 task=7 pos_max=1995",
  ]);
  assert.equal(checkStep("step3", fewAndBroken, p0).verdict, "STOP");
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
// ---- rounds 4 and 5 -----------------------------------------------------------------------------

test("step 3 floor rule: >= 90 % of the ACHIEVABLE reduction, VOID when the gap is too small", () => {
  const withGap = (gap) =>
    `chosen_origin=tolerance chosen_pos_max=1700 gap=${gap} restore_ms=12 reason=tolerance outcome=restored`;
  const p0Of = (gap) =>
    census([F_OFF, ...ARMED, ...Array.from({ length: 6 }, () => restore(withGap(gap)))]);
  const t1 = census([
    F_TAIL,
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
    F_TAIL,
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

test("step 3: T1 recorded with the crosscheck is VOID; a CUDA error is STOP even when VOID", () => {
  const p0 = census([F_OFF, ...ARMED, ...Array.from({ length: 6 }, () => restore(FLAG_OFF))]);
  const xT1 = census([
    F_PROBE,
    ...ARMED,
    ...Array.from({ length: 6 }, () => [
      TAIL_CREATE,
      restore(
        "chosen_origin=tail chosen_pos_max=1995 gap=298 restore_ms=12 reason=tolerance outcome=restored:xcheck-flag-off",
      ),
    ]).flat(),
  ]);
  const r = checkStep("step3", xT1, p0);
  assert.equal(r.verdict, "VOID");
  assert.ok(r.checks.find((c) => /T1 flags/.test(c.name) && !c.ok));
  // CUDA error wins over VOID
  const crashed = census([
    F_TAIL,
    ...ARMED,
    TAIL_CREATE,
    "CUDA error: an illegal memory access was encountered",
  ]);
  assert.equal(checkStep("step3", crashed, p0).verdict, "STOP");
  assert.equal(checkStep("step2", census([...ARMED, "CUDA error: x"])).verdict, "STOP");
});

// raw census state with a flags record, as feedFiles returns it
const rawCensus = (lines) => {
  const c = newCensus();
  for (const l of lines) feed(c, l);
  return c;
};
const armFlags = (a, shell) =>
  a === shell ? F_TAIL : a === "A5" ? F_A5 : a === "A3" ? F_A3 : F_OFF;
const gateCensus = (arms, shell = "C") =>
  Object.fromEntries(arms.map((a) => [a, rawCensus([armFlags(a, shell)])]));

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
  // C recorded with the crosscheck on: mislaunched (VOID), decided from its flags record
  const byArm = gateCensus(["A0", "A1", "C", "A5", "A3"]);
  byArm.C = rawCensus([F_PROBE]);
  const refused = preScoreChecks(records, byArm, { shell: "C" });
  assert.equal(refused.verdict, "mislaunched");
  assert.equal(gateStatus(refused.verdict), "VOID");
});

test("pre-score: flags per arm, CUDA errors in any arm, C HTTP failures STOP, C parse failures drop", () => {
  const ids = ["p0", "p1"];
  const mk = (arm, over = {}) => ({ arm, prompts: ids.map((id) => row(id, over[id] ?? {})) });
  const records = [mk("A0"), mk("A1"), mk("C"), mk("A5"), mk("A3")];
  const arms = ["A0", "A1", "C", "A5", "A3"];
  assert.equal(preScoreChecks(records, gateCensus(arms), { shell: "C" }), null);
  // a flag-off arm recorded with the tail on, or a missing flags record: mislaunched (VOID)
  const tailOnA5 = { ...gateCensus(arms), A5: rawCensus([F_TAIL]) };
  assert.equal(preScoreChecks(records, tailOnA5, { shell: "C" }).verdict, "mislaunched");
  const unknown = { ...gateCensus(arms), A3: newCensus() };
  const u = preScoreChecks(records, unknown, { shell: "C" });
  assert.equal(u.verdict, "mislaunched");
  assert.ok(/flags unknown/.test(u.voidReason));
  // CUDA error lines in any arm: STOP (before the flags check)
  const cuda = {
    ...gateCensus(arms),
    A5: rawCensus([F_OFF, "CUDA error: an illegal memory access was encountered"]),
  };
  const r = preScoreChecks(records, cuda, { shell: "C" });
  assert.equal(r.verdict, "cuda-errors");
  assert.equal(gateStatus(r.verdict), "STOP");
  // an HTTP/connection failure on C where A0 succeeded: STOP
  const cHttp = [
    mk("A0"),
    mk("A1"),
    mk("C", { p1: { ok: false, error: "HTTP 500", errorKind: "http" } }),
    mk("A5"),
    mk("A3"),
  ];
  const r2 = preScoreChecks(cHttp, gateCensus(arms), { shell: "C" });
  assert.equal(r2.verdict, "shell-errors");
  assert.equal(gateStatus(r2.verdict), "STOP");
  // a C response that could not be parsed (UTF-8-split id-count mismatch): not a STOP, a counted drop
  const cParse = [
    mk("A0"),
    mk("A1"),
    mk("C", {
      p1: { ok: false, error: "2 logprobs ids for 3 completion tokens", errorKind: "parse" },
    }),
    mk("A5"),
    mk("A3"),
  ];
  assert.equal(preScoreChecks(cParse, gateCensus(arms), { shell: "C" }), null);
  const pre = engagementAndDrops(
    cParse,
    { A0: ids.map(() => bLine()), A1: ids.map(() => bLine()), C: [tailLine()], A5: [], A3: [] },
    { shell: "C", minPrompts: 1 },
  );
  assert.equal(pre.drop.get("p1"), "C: response not parsed");
  assert.equal(pre.droppedCounts["C: response not parsed"], 1);
});

test("A1's request A differing from A0's is void-determinism, counted before any drop", {
  skip: !fs.existsSync(DEFAULT_LIB),
}, async () => {
  const lib = await import(pathToFileURL(DEFAULT_LIB).href);
  const ids = ["p0", "p1"];
  const mk = (arm, over = {}) => ({
    arm,
    horizon: 8,
    prompts: ids.map((id) => row(id, over[id] ?? {})),
  });
  const benignOver = {
    p0: { continuation: [1, 2, 3, 9, 5, 6, 7, 8] },
    p1: { continuation: [1, 2, 9, 4, 5, 6, 7, 8] },
  };
  // A0 is the odd one out on p1: A1 AND C both differ from it
  const records = [
    mk("A0"),
    mk("A1", { p1: { generated: [10, 11, 8, 13] } }),
    mk("C", { p1: { generated: [10, 11, 8, 13] } }),
    mk("A5", benignOver),
    mk("A3", benignOver),
  ];
  const lines = ids.map(() => bLine());
  const pre = engagementAndDrops(
    records,
    { A0: lines, A1: [lines[0]], C: ids.map(() => tailLine()), A5: [], A3: [] },
    { shell: "C", minPrompts: 1 },
  );
  // C's request-A difference drops p1 ...
  assert.equal(pre.drop.has("p1"), true);
  const s = score(records, lib, {
    shell: "C",
    benign: ["A5", "A3"],
    horizon: 8,
    minPrompts: 1,
    nMin: 1,
    drop: pre.drop,
  });
  // ... but A1's nondeterminism is counted first, so the gate is void-determinism
  assert.equal(s.determinismFailures, 1);
  assert.equal(s.verdict, "void-determinism");
  assert.equal(gateStatus(s.verdict), "VOID");
});
// ---- round 6 ------------------------------------------------------------------------------------

test("B2: C responses unreadable after strict tail restores are C diverging -> STOP (5/24 and 24/24)", () => {
  const ids = Array.from({ length: 24 }, (_, i) => `p${i}`);
  const mk = (arm, over = {}) => ({ arm, prompts: ids.map((id) => row(id, over[id] ?? {})) });
  const parseFail = {
    ok: false,
    error: "23 logprobs ids for 24 completion tokens",
    errorKind: "parse",
  };
  const run = (nBad) => {
    const cOver = Object.fromEntries(ids.slice(0, nBad).map((id) => [id, { ...parseFail }]));
    const records = [mk("A0"), mk("A1"), mk("C", cOver), mk("A5"), mk("A3")];
    const restoresByArm = {
      A0: ids.map(() => bLine()),
      A1: ids.map(() => bLine()),
      C: ids.map(() => tailLine()), // 24 strict tail restores in C's log
      A5: [],
      A3: [],
    };
    const pre = engagementAndDrops(records, restoresByArm, { shell: "C", minPrompts: 20 });
    assert.equal(
      preScoreChecks(records, gateCensus(["A0", "A1", "C", "A5", "A3"]), { shell: "C" }),
      null,
    );
    return pre;
  };
  for (const nBad of [5, 24]) {
    const pre = run(nBad);
    // the parse-failed rows are bound through A0's fields: all 24 tails count as available and used
    assert.equal(pre.tailAvailableAtB, 24);
    assert.equal(pre.tailRestores, 24);
    assert.equal(pre.unreadableAfterTail, nBad);
    assert.equal(pre.slack, 4);
    const v = preVerdict(pre, { shell: "C", minPrompts: 20 });
    assert.equal(v.verdict, "shell-unreadable");
    assert.equal(gateStatus(v.verdict), "STOP");
  }
  // within the slack (4/24): a counted drop, scoring proceeds
  const ok = run(4);
  assert.equal(preVerdict(ok, { shell: "C", minPrompts: 20 }), null);
  assert.equal(ok.droppedCounts["C: response not parsed"], 4);
});

test("steps 1 and 2 are VOID when the log shows the server never served", () => {
  const noServe = [
    ...ARMED.filter((l) => l !== LISTENING),
    TAIL_CREATE,
    restore(TAIL_HIT),
    XCHECK_ROW,
  ];
  assert.equal(checkStep("step2", census([F_PROBE, ...noServe])).verdict, "VOID");
  assert.equal(checkStep("step1", census([F_OFF, ...noServe])).verdict, "VOID");
  // another server held the port: bind failure
  const bound = [
    ...ARMED,
    "couldn't bind to server socket: hostname=0.0.0.0 port=8099",
    TAIL_CREATE,
    restore(TAIL_HIT),
    XCHECK_ROW,
  ];
  const r = checkStep("step2", census([F_PROBE, ...bound]));
  assert.equal(r.verdict, "VOID");
  assert.ok(r.checks.find((c) => /server served/.test(c.name) && !c.ok));
  // a CUDA error still wins (STOP)
  assert.equal(checkStep("step2", census([F_PROBE, ...noServe, "CUDA error: x"])).verdict, "STOP");
});

test("step 4: -ExtraArgs of the benign arms (and of C) are checked against the required values", () => {
  const ids = ["p0"];
  const mk = (arm) => ({ arm, prompts: ids.map((id) => row(id)) });
  const records = ["A0", "A1", "C", "A5", "A3"].map(mk);
  const arms = ["A0", "A1", "C", "A5", "A3"];
  assert.equal(preScoreChecks(records, gateCensus(arms), { shell: "C" }), null);
  // benign arms launched without their perturbation: identical to A0 -> mislaunched (VOID)
  const noA5 = { ...gateCensus(arms), A5: rawCensus([F_OFF]) };
  const r = preScoreChecks(records, noA5, { shell: "C" });
  assert.equal(r.verdict, "mislaunched");
  assert.ok(/extra=''/.test(r.voidReason));
  const wrongA3 = { ...gateCensus(arms), A3: rawCensus([flags(1, 0, 0, "-fa 1")]) };
  assert.equal(preScoreChecks(records, wrongA3, { shell: "C" }).verdict, "mislaunched");
  // C launched with -fa 0 is mislaunched too
  const cFa0 = { ...gateCensus(arms), C: rawCensus([flags(1, 1, 0, "-fa 0")]) };
  assert.equal(preScoreChecks(records, cFa0, { shell: "C" }).verdict, "mislaunched");
  // whitespace in the recorded extra does not matter
  const spaced = { ...gateCensus(arms), A5: rawCensus([flags(1, 0, 0, "  -no-fmoe   -no-fug ")]) };
  assert.equal(preScoreChecks(records, spaced, { shell: "C" }), null);
});

test("per-log flags records: every log needs its own, all must agree; sidecar vs header conflict is VOID", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "stateos-flags-"));
  try {
    const writeLog = (stem, flagsLine, body = [LISTENING]) => {
      fs.writeFileSync(path.join(dir, `${stem}.err.log`), `${body.join("\n")}\n`);
      if (flagsLine) fs.writeFileSync(path.join(dir, `${stem}.flags`), `${flagsLine}\n`);
      return path.join(dir, `${stem}.err.log`);
    };
    const probe = writeLog("probe", F_PROBE, ARMED);
    const div = writeLog("div", F_OFF, ARMED);
    const bare = writeLog("bare", null, ARMED);
    // one file's flags never stand in for another's
    const mixed = summarize(await feedFiles([probe, div]));
    assert.equal(mixed.flags, null);
    assert.ok(/inconsistent across logs/.test(mixed.flagsProblem));
    assert.equal(checkStep("step1", mixed).verdict, "VOID");
    const missing = summarize(await feedFiles([div, bare]));
    assert.ok(/flags unknown for/.test(missing.flagsProblem));
    assert.equal(checkStep("step1", missing).verdict, "VOID");
    // two logs with the same record: accepted
    const div2 = writeLog("div2", F_OFF, ARMED);
    const same = summarize(await feedFiles([div, div2]));
    assert.equal(same.flagsProblem, null);
    assert.equal(same.flags.LONGSPEAR_STATEOS_DIV_LOG, "1");
    // a header line in the log that differs from the sidecar
    const conflict = writeLog("conflict", F_OFF, [F_PROBE, ...ARMED]);
    const c = summarize(await feedFiles([conflict]));
    assert.ok(/inconsistent within/.test(c.flagsProblem));
    assert.equal(checkStep("step1", c).verdict, "VOID");
    // the server's listening line is read from <LogStem>.out.log
    const quiet = writeLog(
      "quiet",
      F_OFF,
      ARMED.filter((l) => l !== LISTENING),
    );
    assert.equal(summarize(await feedFiles([quiet])).served.ok, false);
    fs.writeFileSync(path.join(dir, "quiet.out.log"), `${LISTENING}\n`);
    assert.equal(summarize(await feedFiles([quiet])).served.ok, true);
    // a log with no sidecar at all: flags unknown (e.g. a server not started by the launcher)
    const alone = summarize(await feedFiles([bare]));
    assert.equal(checkStep("step1", alone).verdict, "VOID");
    assert.ok(checkStep("step1", alone).checks.find((x) => /flags unknown/.test(x.detail)));
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("step 3: a CUDA error only in the P0 log is STOP, even when a VOID check fails", () => {
  const p0 = census([
    F_OFF,
    ...ARMED,
    restore(FLAG_OFF),
    "CUDA error: an illegal memory access was encountered",
  ]);
  const t1 = census([F_TAIL, ...ARMED, TAIL_CREATE, restore(TAIL_HIT)]); // too few events: VOID checks fail
  const r = checkStep("step3", t1, p0);
  assert.equal(r.verdict, "STOP");
  assert.ok(r.checks.find((c) => /P0 CUDA/.test(c.name) && !c.ok));
});
