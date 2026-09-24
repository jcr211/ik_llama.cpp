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
  portRecordProblem,
  setPortRecord,
  summarize,
} from "./stateos-div-census.mjs";
import {
  DEFAULT_LIB,
  engagementAndDrops,
  forcedPrompt,
  gateRefusal,
  gateStatus,
  rowProblem,
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

// the server's startup line (normally from <LogStem>.out.log), the chain's port check (normally the
// <LogStem>.port sidecar) and an armed PLE-history repair
const LISTENING =
  'INFO [                    main] HTTP server listening | tid="1" timestamp=1 hostname="0.0.0.0" port="8099"';
// the <LogStem>.port / <LogStem>.pid contents of a correctly launched server (the only port evidence)
const PORT_OK = "[stateos-port] ok pid=1 listeners=1 port=8099";
const PID = "1";
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

// port = [<LogStem>.port text, <LogStem>.pid text] as feedFiles reads them (null = no record)
const census = (lines, port = [PORT_OK, PID]) => {
  const c = newCensus();
  for (const l of lines) feed(c, l);
  if (port) setPortRecord(c, port[0], port[1]);
  return summarize(c);
};

// the launcher's recorded flags ([stateos-flags], normally from the <LogStem>.flags sidecar)
const flags = (div, tail, xcheck, extra = "", stem = "x", spec = "on") =>
  `[stateos-flags] LONGSPEAR_STATEOS_DIV_LOG=${div} LONGSPEAR_STATEOS_TAIL_SNAPSHOT=${tail} LONGSPEAR_STATEOS_TAIL_XCHECK=${xcheck} LONGSPEAR_PLE_HIST_REWIND=1 LONGSPEAR_PLE_HIST_LOG=1 spec=${spec} logstem=${stem} extra='${extra}'`;
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
// record-level fields of one consistent run (checkRecords); horizon 8 = row()'s continuation length
const recMeta = (arm) => ({
  arm,
  horizon: 8,
  nFirst: 64,
  receipt: "receipt.json",
  url: "http://127.0.0.1:8099",
  logStem: `stem-${arm}`,
});
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
// a request-B restore line; by default the final round of request A accepted drafts (eligible in A0)
const bLine = (over = {}) => ({
  nPast: 5,
  tailDist: 1,
  prevRound: "drafted",
  prevNAcc: 2,
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
  const mk = (arm, over = {}) => ({
    ...recMeta(arm),
    prompts: ids.map((id) => row(id, over[id] ?? {})),
  });
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
  assert.equal(r.tailRestores, 2);
  assert.equal(r.eligible, 4);
  assert.equal(r.engaged, true);
  assert.deepEqual([...r.drop.keys()].sort(), ["p2", "p3"]);
  assert.equal(r.droppedCounts["C: request A output differs from A0's"], 1);
  assert.equal(r.droppedCounts["A3: request B differs from A0's"], 1);
  // C's request-A difference counts against the shared slack; A3's different B does not
  assert.equal(r.cExcluded, 1);
  assert.deepEqual(r.cExclusions, [
    { id: "p2", request: "A", reason: "request A output differs from A0's" },
  ]);
  assert.equal(r.nonCExcluded, 1);
  assert.equal(preVerdict(r, { shell: "C", minPrompts: 2 }), null);
  // C did not restore from the tail on prompts A0 shows as eligible: C's fault (3 > slack 2) -> STOP
  const unused = engagementAndDrops(
    records,
    { ...restoresByArm, C: ids.map(() => bLine({ tailAvailable: true })) },
    { shell: "C", minPrompts: 2 },
  );
  assert.equal(unused.cExcluded, 3);
  assert.equal(unused.cReasons["not a strict tail restore"], 2);
  assert.equal(preVerdict(unused, { shell: "C", minPrompts: 2 }).verdict, "shell-diverged");
  assert.equal(gateStatus("not-engaged:no-eligible-prompts"), "VOID");
  assert.equal(gateStatus("shell-diverged"), "STOP");
  assert.equal(gateStatus("ple-hist"), "STOP");
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
      generated: [10, 11, 7, 13], // a valid request A (rowProblem needs >= 2 tokens)
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

// raw census state with a flags record and a port record, as feedFiles returns it
const rawCensus = (lines, port = [PORT_OK, PID]) => {
  const c = newCensus();
  for (const l of lines) feed(c, l);
  if (port) setPortRecord(c, port[0], port[1]);
  return c;
};
// each arm's recorded flags, with its own logstem (recMeta's logStem) and speculation state
const armFlags = (a, shell, spec = "on") =>
  a === shell
    ? flags(1, 1, 0, "", `stem-${a}`, spec)
    : a === "A5"
      ? flags(1, 0, 0, "-no-fmoe -no-fug", `stem-${a}`, spec)
      : a === "A3"
        ? flags(1, 0, 0, "-fa 0", `stem-${a}`, spec)
        : flags(1, 0, 0, "", `stem-${a}`, spec);
// every arm: its recorded flags, the port check and an armed PLE-history repair
const gateCensus = (arms, shell = "C", spec = "on") =>
  Object.fromEntries(arms.map((a) => [a, rawCensus([armFlags(a, shell, spec), ...ARMED])]));

test("Opus repro: a C arm served by the crosscheck never scores (restored:xcheck-flag-off is no tail restore)", () => {
  const ids = Array.from({ length: 24 }, (_, i) => `p${i}`);
  const mk = (arm) => ({ ...recMeta(arm), prompts: ids.map((id) => row(id)) });
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
  const mk = (arm, over = {}) => ({
    ...recMeta(arm),
    prompts: ids.map((id) => row(id, over[id] ?? {})),
  });
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
  // C's HTTP and parse failures are not log checks: they are C-attributable exclusions (shared slack)
  const cFail = [
    mk("A0"),
    mk("A1"),
    mk("C", {
      p0: { ok: false, error: "HTTP 500", errorKind: "http", failedAt: "B" },
      p1: {
        ok: false,
        error: "2 logprobs ids for 3 completion tokens",
        errorKind: "parse",
        failedAt: "A",
      },
    }),
    mk("A5"),
    mk("A3"),
  ];
  assert.equal(preScoreChecks(cFail, gateCensus(arms), { shell: "C" }), null);
  const lines = {
    A0: ids.map(() => bLine()),
    A1: ids.map(() => bLine()),
    C: [tailLine()],
    A5: [],
    A3: [],
  };
  const pre = engagementAndDrops(cFail, lines, { shell: "C", minPrompts: 1 });
  assert.deepEqual(
    pre.cExclusions.map((e) => [e.id, e.request, e.reason]),
    [
      ["p0", "B", "request B HTTP/connection error"],
      ["p1", "A", "request A not parsed"],
    ],
  );
  // 2 > slack 1: STOP
  assert.equal(preVerdict(pre, { shell: "C", minPrompts: 1 }).verdict, "shell-diverged");
  // one of them (1 <= slack 1): a counted drop
  const one = engagementAndDrops(
    [mk("A0"), mk("A1"), mk("C", { p1: cFail[2].prompts[1] }), mk("A5"), mk("A3")],
    lines,
    {
      shell: "C",
      minPrompts: 1,
    },
  );
  assert.equal(preVerdict(one, { shell: "C", minPrompts: 1 }), null);
  assert.equal(one.droppedCounts["C: request A not parsed"], 1);
});

test("N10/N8: step 4 PLE checks (STOP) and the port check (VOID) per arm", () => {
  const ids = ["p0"];
  const records = ["A0", "A1", "C", "A5", "A3"].map((arm) => ({
    ...recMeta(arm),
    prompts: ids.map((id) => row(id)),
  }));
  const arms = ["A0", "A1", "C", "A5", "A3"];
  const noPle = ARMED.filter((l) => !l.startsWith("[ple-hist] set"));
  // no [ple-hist] set in C's log although its flags record REWIND=1 LOG=1: STOP
  const unarmed = { ...gateCensus(arms), C: rawCensus([F_TAIL, ...noPle]) };
  const r = preScoreChecks(records, unarmed, { shell: "C" });
  assert.equal(r.verdict, "ple-hist");
  assert.equal(gateStatus(r.verdict), "STOP");
  // an unrepaired rewind in a benign arm: STOP
  const rewind = {
    ...gateCensus(arms),
    A3: rawCensus([F_A3, ...ARMED, "[ple-hist] reset seq=0 pos=1996 next_pos=2001"]),
  };
  assert.equal(preScoreChecks(records, rewind, { shell: "C" }).verdict, "ple-hist");
  // a second listener on the port, or no port record: mislaunched (VOID)
  const shared = {
    ...gateCensus(arms),
    A1: rawCensus([F_OFF, ...ARMED], ["[stateos-port] shared pid=7 listeners=7,9 port=8099", "7"]),
  };
  const s = preScoreChecks(records, shared, { shell: "C" });
  assert.equal(s.verdict, "mislaunched");
  assert.ok(/A1: mislaunched: port shared/.test(s.voidReason));
  assert.equal(gateStatus(s.verdict), "VOID");
  const unverified = { ...gateCensus(arms), C: rawCensus([F_TAIL, ...ARMED], null) };
  assert.ok(
    /C: port unverified/.test(preScoreChecks(records, unverified, { shell: "C" }).voidReason),
  );
  // the census steps apply the same port rule
  const s2 = census([F_PROBE, ...ARMED, TAIL_CREATE, restore(TAIL_HIT), XCHECK_ROW], null);
  assert.equal(checkStep("step2", s2).verdict, "VOID");
});

test("round 8 (Sol nit): port evidence is ONLY the .port sidecar, validated against .pid", async () => {
  // contents: ok + model port + exactly one listener == the launched PID; anything else is a problem
  assert.equal(portRecordProblem(PORT_OK, PID), null);
  assert.equal(portRecordProblem(`${PORT_OK}\r\n`, "1\r\n"), null);
  for (const [portText, pidText] of [
    ["[stateos-port] ok pid=1 listeners=1 port=8101", "1"], // another port
    ["[stateos-port] ok pid=1 listeners=1,2 port=8099", "1"], // two listeners
    ["[stateos-port] ok pid=1 listeners=2 port=8099", "1"], // the listener is not the launched PID
    ["[stateos-port] ok pid=1 listeners=1 port=8099", "5"], // .pid says another process
    ["[stateos-port] ok pid=1 listeners=1 port=8099", null], // no .pid
    [null, "1"], // no .port
    ["[stateos-port] ok", "1"], // malformed
    [`${PORT_OK}\n${PORT_OK}`, "1"], // more than one record
  ]) {
    assert.ok(portRecordProblem(portText, pidText), JSON.stringify([portText, pidText]));
  }
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "stateos-port-"));
  try {
    const log = (stem, body, port, pid) => {
      const f = path.join(dir, `${stem}.err.log`);
      fs.writeFileSync(f, `${body.join("\n")}\n`);
      fs.writeFileSync(path.join(dir, `${stem}.flags`), `${F_TAIL}\n`);
      if (port !== null) fs.writeFileSync(path.join(dir, `${stem}.port`), `${port}\n`);
      if (pid !== null) fs.writeFileSync(path.join(dir, `${stem}.pid`), `${pid}\n`);
      return f;
    };
    const good = await feedFiles([log("good", ARMED, PORT_OK, PID)]);
    assert.equal(summarize(good).portProblem, null);
    // an ok line inside the LOG is never port evidence, with or without a failing sidecar
    const inLog = await feedFiles([log("inlog", [...ARMED, PORT_OK], null, PID)]);
    assert.ok(/port unverified/.test(summarize(inLog).portProblem));
    const masked = await feedFiles([
      log(
        "masked",
        [...ARMED, PORT_OK],
        "[stateos-port] shared pid=1 listeners=1,9 port=8099",
        PID,
      ),
    ]);
    assert.ok(/port shared/.test(summarize(masked).portProblem));
    const wrongPid = await feedFiles([log("wrongpid", ARMED, PORT_OK, "77")]);
    assert.ok(/launched 77/.test(summarize(wrongPid).portProblem));
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// ---- round 7: one shared slack for every exclusion attributable to C ----------------------------

const ids24 = Array.from({ length: 24 }, (_, i) => `p${i}`);
const mk24 = (arm, over = {}) => ({
  ...recMeta(arm),
  prompts: ids24.map((id) => row(id, over[id] ?? {})),
});
const lines24 = (c = ids24.map(() => tailLine())) => ({
  A0: ids24.map(() => bLine()),
  A1: ids24.map(() => bLine()),
  C: c,
  A5: [],
  A3: [],
});
const gate24 = (cOver, restores = lines24()) => {
  const records = [mk24("A0"), mk24("A1"), mk24("C", cOver), mk24("A5"), mk24("A3")];
  assert.equal(
    preScoreChecks(records, gateCensus(["A0", "A1", "C", "A5", "A3"]), { shell: "C" }),
    null,
  );
  const pre = engagementAndDrops(records, restores, { shell: "C", minPrompts: 20 });
  return { pre, v: preVerdict(pre, { shell: "C", minPrompts: 20 }) };
};

test("round 7 P1: B unreadable after a strict tail restore on p0, A unreadable on p1-p23 -> STOP", () => {
  const parse = (failedAt) => ({
    ok: false,
    error: "23 logprobs ids for 24 completion tokens",
    errorKind: "parse",
    failedAt,
  });
  const cOver = Object.fromEntries(ids24.map((id, i) => [id, parse(i === 0 ? "B" : "A")]));
  // C's log: one strict tail restore (p0's B); requests A of p1..p23 never reached a B
  const { pre, v } = gate24(cOver, lines24([tailLine()]));
  assert.equal(pre.slack, 4);
  assert.equal(pre.cExcluded, 24);
  assert.deepEqual(pre.cReasons, { "request B not parsed": 1, "request A not parsed": 23 });
  assert.equal(pre.cExclusions[0].request, "B");
  assert.equal(v.verdict, "shell-diverged");
  assert.equal(gateStatus(v.verdict), "STOP");
  assert.ok(/request A not parsed: 23/.test(v.voidReason));
});

test("round 7 P2: parseable garbage (C's request A differs) on p1-p23 -> STOP, never VOID", () => {
  const cOver = Object.fromEntries(
    ids24.map((id, i) =>
      i === 0
        ? [id, { ok: false, error: "garbled", errorKind: "parse", failedAt: "B" }]
        : [id, { generated: [42, 42, 42, 42] }],
    ),
  );
  const { pre, v } = gate24(cOver, lines24([tailLine()]));
  assert.equal(pre.cExcluded, 24);
  assert.equal(pre.cReasons["request A output differs from A0's"], 23);
  assert.equal(v.verdict, "shell-diverged");
  assert.equal(gateStatus(v.verdict), "STOP");
});

test("round 7 P3: C's request A differs on 5 of 24, 19 clean strict tail restores -> STOP (5 > 4)", () => {
  const cOver = Object.fromEntries(
    ids24.slice(0, 5).map((id) => [id, { generated: [42, 42, 42, 42] }]),
  );
  // C still sends A0's B on every prompt, so its log has a request-B line for each of the 24
  const { pre, v } = gate24(cOver);
  assert.equal(pre.tailRestores, 19);
  assert.equal(pre.cExcluded, 5);
  assert.equal(v.verdict, "shell-diverged");
  // 4 of 24: tolerated by design, reported with the reason, then scored
  const four = gate24(
    Object.fromEntries(ids24.slice(0, 4).map((id) => [id, { generated: [42, 42, 42, 42] }])),
  );
  assert.equal(four.v, null);
  assert.equal(four.pre.cExcluded, 4);
  assert.equal(four.pre.tailRestores, 20);
  assert.equal(four.pre.droppedCounts["C: request A output differs from A0's"], 4);
});

test("round 7: a genuine no-eligible-prompts case (A0 shows no accepted draft) is VOID", () => {
  const ineligible = () => bLine({ prevRound: "root-only", prevNAcc: 0 });
  const records = [mk24("A0"), mk24("A1"), mk24("C"), mk24("A5"), mk24("A3")];
  const restores = {
    ...lines24(ids24.map(() => bLine())),
    A0: ids24.map(ineligible),
    A1: ids24.map(ineligible),
  };
  const pre = engagementAndDrops(records, restores, { shell: "C", minPrompts: 20 });
  assert.equal(pre.eligible, 0);
  assert.equal(pre.cExcluded, 0);
  assert.equal(pre.nonCExcluded, 24);
  const v = preVerdict(pre, { shell: "C", minPrompts: 20 });
  assert.equal(v.verdict, "not-engaged:no-eligible-prompts");
  assert.equal(gateStatus(v.verdict), "VOID");
  // but the same C behaviour where A0 shows eligible prompts is C's fault: STOP
  const eligibleCase = engagementAndDrops(records, lines24(ids24.map(() => bLine())), {
    shell: "C",
    minPrompts: 20,
  });
  assert.equal(eligibleCase.cExcluded, 24);
  assert.equal(preVerdict(eligibleCase, { shell: "C", minPrompts: 20 }).verdict, "shell-diverged");
});

// ---- round 8: determinism before C attribution, control failures, one validity rule, all arms ----

const ARMS = ["A0", "A1", "C", "A5", "A3"];
const OPTS = { shell: "C", benign: ["A5", "A3"], minPrompts: 20 };
const refuse24 = (over = {}, restores = lines24()) =>
  gateRefusal(
    ARMS.map((a) => mk24(a, over[a] ?? {})),
    gateCensus(ARMS),
    restores,
    OPTS,
  );
const onIds = (ids, o) => Object.fromEntries(ids.map((id) => [id, { ...o }]));
const OTHER_A = { generated: [42, 42, 42, 42] };
const OTHER_B = { continuation: [1, 2, 3, 9, 9, 9, 9, 9] };

test("round 8 (Opus S1/S1b/S3): A1 != A0 is void-determinism BEFORE any C attribution", () => {
  // control: all clean -> scored
  assert.equal(refuse24().refused, null);
  // S1: A1's request A differs on p0, C's request A differs on p1-p5 (5 > slack 4)
  const s1 = refuse24({ A1: { p0: OTHER_A }, C: onIds(ids24.slice(1, 6), OTHER_A) }).refused;
  assert.equal(s1.verdict, "void-determinism");
  assert.equal(gateStatus(s1.verdict), "VOID");
  assert.ok(/p0@A/.test(s1.voidReason));
  // S1b: A1's B continuation differs on p0 (its A matches), C's request A differs on p1-p5
  const s1b = refuse24({ A1: { p0: OTHER_B }, C: onIds(ids24.slice(1, 6), OTHER_A) }).refused;
  assert.equal(s1b.verdict, "void-determinism");
  assert.ok(/p0@B/.test(s1b.voidReason));
  // S3: A1's B continuation differs on p0 and C's request A differs on p0 (a drop would hide it)
  const s3 = refuse24({ A1: { p0: OTHER_B }, C: { p0: OTHER_A } }).refused;
  assert.equal(s3.verdict, "void-determinism");
  // hard checks still come first: a CUDA error in any arm is STOP even with A1 != A0
  const cuda = gateRefusal(
    ARMS.map((a) => mk24(a, a === "A1" ? { p0: OTHER_A } : {})),
    { ...gateCensus(ARMS), C: rawCensus([armFlags("C", "C"), ...ARMED, "CUDA error: x"]) },
    lines24(),
    OPTS,
  ).refused;
  assert.equal(cuda.verdict, "cuda-errors");
});

test("round 8 (Sol bug 2): an invalid A1 row is a control failure, never charged to C", () => {
  const http = { ok: false, error: "fetch failed", errorKind: "http", failedAt: "A" };
  // a common outage: A1 AND C fail request A on all 24 prompts -> VOID (insufficient control), not STOP
  const outage = refuse24({ A1: onIds(ids24, http), C: onIds(ids24, http) });
  assert.equal(outage.pre.cExcluded, 0);
  assert.equal(outage.pre.controlExcluded, 24);
  assert.equal(outage.refused.verdict, "insufficient-control");
  assert.equal(gateStatus(outage.refused.verdict), "VOID");
  // an A1 parse failure on 3 prompts where C also failed: not C's; the other 21 score
  const parse = { ok: false, error: "x", errorKind: "parse", failedAt: "B" };
  const some = refuse24({ A1: onIds(ids24.slice(0, 3), parse), C: onIds(ids24.slice(0, 3), http) });
  assert.equal(some.pre.cExcluded, 0);
  assert.equal(some.refused, null);
});

test("round 8 (Sol bug 1): zero-token responses are invalid by the one rule shared with score", {
  skip: !fs.existsSync(DEFAULT_LIB),
}, async () => {
  const lib = await import(pathToFileURL(DEFAULT_LIB).href);
  // C returns 0 tokens on all 24 eligible B prompts (with strict tail restores in its log) -> STOP
  const empty = refuse24({ C: onIds(ids24, { continuation: [] }) });
  assert.equal(empty.pre.cExcluded, 24);
  assert.deepEqual(empty.pre.cReasons, { "request B returned fewer tokens than the horizon": 24 });
  assert.equal(empty.refused.verdict, "shell-diverged");
  assert.equal(gateStatus(empty.refused.verdict), "STOP");
  // a 1-token request A from C is invalid too
  assert.equal(rowProblem(row("p0", { generated: [10] }), 8).request, "A");
  // A0/A1 returning 0 tokens is a control failure
  const a0Empty = refuse24({ A0: onIds(ids24, { continuation: [] }) });
  assert.equal(a0Empty.pre.cExcluded, 0);
  assert.equal(a0Empty.refused.verdict, "insufficient-control");
  // within the slack: engagement's drops and score agree; score never rejects a row engagement scored
  const four = refuse24({ C: onIds(ids24.slice(0, 4), { continuation: [] }) });
  assert.equal(four.refused, null);
  const records = ARMS.map((a) =>
    mk24(
      a,
      a === "C"
        ? onIds(ids24.slice(0, 4), { continuation: [] })
        : a === "A5" || a === "A3"
          ? onIds(ids24, OTHER_B)
          : {},
    ),
  ).map((r) => ({ ...r, horizon: 8 }));
  const s = score(records, lib, { ...OPTS, horizon: 8, nMin: 1, drop: four.pre.drop });
  assert.equal(s.rows.length, 20);
  assert.equal(s.dropped.filter((d) => /invalid/.test(d.reason)).length, 0);
});

test("round 8 (Opus nit): refuse to score unless A0, A1, C, A5 and A3 records all exist", () => {
  for (const missing of ARMS) {
    const records = ARMS.filter((a) => a !== missing).map((a) => mk24(a));
    assert.throws(
      () => gateRefusal(records, gateCensus(ARMS), lines24(), OPTS),
      new RegExp(`missing arm record\\(s\\) arm-${missing}\\.json`),
    );
  }
});

// ---- round 9: one consistent run, horizon-length continuations, spec-off fallback, precedence ------

const recs24 = (over = {}) => ARMS.map((a) => mk24(a, over[a] ?? {}));
const rec = (records, arm) => records.find((r) => r.arm === arm);

test("round 9 (Opus E1: S1, S2, S2b, S3): inconsistent arm records are refused (exit 1)", () => {
  const refuse = (records, census = gateCensus(ARMS)) =>
    assert.throws(() => gateRefusal(records, census, lines24(), OPTS), /refusing to score/);
  // S1: a stale second A0 record (every row failed) next to the real one
  const stale = { ...mk24("A0", onIds(ids24, { ok: false, error: "x", errorKind: "http" })) };
  refuse([stale, ...recs24()]);
  // S2: C run with a shorter horizon than the others; S2b: the benign arms with a shorter horizon
  const s2 = recs24();
  rec(s2, "C").horizon = 4;
  refuse(s2);
  const s2b = recs24();
  rec(s2b, "A5").horizon = 4;
  rec(s2b, "A3").horizon = 4;
  refuse(s2b);
  // S3: C's record from a --limit 20 run
  const s3 = recs24();
  rec(s3, "C").prompts = rec(s3, "C").prompts.slice(0, 20);
  refuse(s3);
  // nFirst, receipt, url port, an extra arm, a record tied to another log
  for (const mutate of [
    (r) => (rec(r, "A1").nFirst = 32),
    (r) => (rec(r, "A3").receipt = "other.json"),
    (r) => (rec(r, "A5").url = "http://127.0.0.1:8101"),
    (r) => r.push({ ...mk24("A0"), arm: "A0-attempt1" }),
    (r) => (rec(r, "A5").logStem = "stem-A0"), // N3: A5's record must name A5's own server log
  ]) {
    const r = recs24();
    mutate(r);
    refuse(r);
  }
  // a consistent run passes the record check
  assert.equal(gateRefusal(recs24(), gateCensus(ARMS), lines24(), OPTS).refused, null);
});

test("round 9 (Sol): control failures beyond the slack win over C's exclusions (VOID, not STOP)", () => {
  const http = { ok: false, error: "x", errorKind: "http", failedAt: "A" };
  const r = refuse24({
    A1: onIds(ids24.slice(0, 5), http),
    C: onIds(ids24.slice(10, 15), OTHER_A),
  });
  assert.equal(r.pre.controlExcluded, 5);
  assert.equal(r.pre.cExcluded, 5);
  assert.equal(r.refused.verdict, "insufficient-control");
  assert.equal(gateStatus(r.refused.verdict), "VOID");
});

test("round 9 (Sol): a B continuation must reach the horizon; EOS before it is invalid for every arm", {
  skip: !fs.existsSync(DEFAULT_LIB),
}, async () => {
  const lib = await import(pathToFileURL(DEFAULT_LIB).href);
  // 24 identical one-token B responses in every arm: never compatible-at-horizon
  const short = Object.fromEntries(ARMS.map((a) => [a, onIds(ids24, { continuation: [5] })]));
  const g = refuse24(short);
  assert.equal(g.refused.verdict, "insufficient-control");
  const s = score(recs24(short), lib, { ...OPTS, horizon: 8, nMin: 1 });
  assert.notEqual(s.verdict, "compatible-at-horizon");
  // an EOS end is named as such, and is invalid (the v2 lib cannot express a genuine end)
  const eos = rowProblem(row("p0", { continuation: [1, 2, 3], eosAt: 3, bFinish: "stop" }), 8);
  assert.equal(eos.request, "B");
  assert.equal(eos.reason, "request B ended on EOS before the horizon");
  assert.equal(rowProblem(row("p0"), 8), null);
});

test("round 9 (Opus N1/S4): A1's request-B failure is a control failure; its request A still counts for determinism", () => {
  const bFail = { ok: false, error: "HTTP 500", errorKind: "http", failedAt: "B" };
  // A1's B fails on p0: excluded as a control failure, never a C exclusion (even if C also failed)
  const one = refuse24({ A1: { p0: bFail }, C: { p0: bFail } });
  assert.equal(one.pre.cExcluded, 0);
  assert.equal(one.pre.controlExcluded, 1);
  assert.equal(one.refused, null);
  // S4: A1's request A differs on p0 and then its B fails; C's request A differs on p1-p5
  const s4 = refuse24({
    A1: { p0: { ...bFail, generated: [42, 42, 42, 42] } },
    C: onIds(ids24.slice(1, 6), OTHER_A),
  });
  assert.equal(s4.refused.verdict, "void-determinism");
});

test("round 9 (Opus N4): the spec-off fallback runs when EVERY arm is spec=off; mixed = mislaunched", () => {
  const off = gateRefusal(recs24(), gateCensus(ARMS, "C", "off"), lines24(), OPTS);
  assert.equal(off.specOff, true);
  assert.equal(off.refused, null);
  assert.equal(gateRefusal(recs24(), gateCensus(ARMS), lines24(), OPTS).specOff, false);
  const mixed = { ...gateCensus(ARMS), A3: rawCensus([armFlags("A3", "C", "off"), ...ARMED]) };
  const m = gateRefusal(recs24(), mixed, lines24(), OPTS).refused;
  assert.equal(m.verdict, "mislaunched");
  assert.ok(/speculation states differ/.test(m.voidReason));
  // a flags record without spec= is "unknown": mislaunched in step 4, and steps 1-3 need spec=on
  const noSpec = armFlags("A1", "C").replace(" spec=on", "");
  const u = gateRefusal(
    recs24(),
    { ...gateCensus(ARMS), A1: rawCensus([noSpec, ...ARMED]) },
    lines24(),
    OPTS,
  );
  assert.equal(u.refused.verdict, "mislaunched");
  assert.equal(
    checkStep("step1", census([flags(1, 0, 0, "", "x", "off"), ...ARMED])).verdict,
    "VOID",
  );
});

test("round 9: verdict precedence, one case per adjacent pair of rules (the first rule wins)", () => {
  const setRow = (s, arm, i, over) => {
    rec(s.records, arm).prompts[i] = row(ids24[i], over);
  };
  // each rule's injection touches its own arm / prompts so any two can be combined
  const RULES = [
    ["records", (s) => (rec(s.records, "A5").horizon = 4), "throw", null],
    ["cuda", (s) => feed(s.census.A0, "CUDA error: x"), "cuda-errors", null],
    [
      "flags",
      (s) => (s.census.A5.flags = { ...s.census.A5.flags, extra: "" }),
      "mislaunched",
      /recorded launch flags/,
    ],
    [
      "ple-reset",
      (s) => feed(s.census.A3, "[ple-hist] reset seq=0 pos=1996 next_pos=2001"),
      "ple-hist",
      /unrepaired/,
    ],
    ["port", (s) => setPortRecord(s.census.A1, null, PID), "mislaunched", /port ownership/],
    ["ple-set", (s) => (s.census.C.ple.sets = 0), "ple-hist", /never logged a set/],
    ["determinism", (s) => setRow(s, "A1", 0, OTHER_A), "void-determinism", null],
    [
      "control",
      (s) => {
        for (const i of [10, 11, 12, 13, 14])
          setRow(s, "A1", i, { ok: false, error: "x", errorKind: "http", failedAt: "B" });
      },
      "insufficient-control",
      null,
    ],
    [
      "c-slack",
      (s) => {
        for (const i of [15, 16, 17, 18, 19]) setRow(s, "C", i, OTHER_A);
      },
      "shell-diverged",
      null,
    ],
    [
      "not-engaged",
      (s) => {
        for (const i of [5, 6, 7, 8, 9]) {
          s.restores.A0[i] = bLine({ prevRound: "root-only", prevNAcc: 0 });
          s.restores.C[i] = bLine();
        }
      },
      "not-engaged:no-eligible-prompts",
      null,
    ],
    ["score", () => {}, null, null],
  ];
  const run = (injectors) => {
    const s = { records: recs24(), census: gateCensus(ARMS), restores: lines24() };
    for (const inject of injectors) inject(s);
    return gateRefusal(s.records, s.census, s.restores, OPTS).refused;
  };
  const expectRule = ([name, , verdict, re], got, label) => {
    if (verdict === null) return assert.equal(got, null, `${label}: expected scoring`);
    assert.equal(got?.verdict, verdict, `${label}: expected ${name}`);
    if (re) assert.ok(re.test(got.voidReason), `${label}: ${got.voidReason}`);
  };
  for (let i = 0; i < RULES.length; i += 1) {
    const label = i + 1 < RULES.length ? `${RULES[i][0]} + ${RULES[i + 1][0]}` : RULES[i][0];
    const injectors = i + 1 < RULES.length ? [RULES[i][1], RULES[i + 1][1]] : [RULES[i][1]];
    if (RULES[i][2] === "throw") {
      assert.throws(() => run(injectors), /refusing to score/, label);
      continue;
    }
    expectRule(RULES[i], run(injectors), label);
    // and each rule alone fires its own verdict
    expectRule(RULES[i], run([RULES[i][1]]), `${RULES[i][0]} alone`);
  }
});

test("N9: feedFiles resets per-log parser state (a pending tail at the end of log 1 never reaches log 2)", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "stateos-reset-"));
  try {
    const write = (stem, body) => {
      fs.writeFileSync(path.join(dir, `${stem}.err.log`), `${body.join("\n")}\n`);
      fs.writeFileSync(path.join(dir, `${stem}.flags`), `${F_TAIL}\n`);
      return path.join(dir, `${stem}.err.log`);
    };
    // log 1 ends right after a tail create (pending); log 2 starts with a restore and no create
    const log1 = write("one", [...ARMED, TAIL_CREATE]);
    const log2 = write("two", [...ARMED, restore(FLAG_OFF)]);
    const c = await feedFiles([log1, log2]);
    assert.equal(c.v2.restores.length, 1);
    assert.equal(c.v2.restores[0].tailAvailable, false);
    // the same lines in ONE log: the tail is pending at the restore
    const both = await feedFiles([write("both", [...ARMED, TAIL_CREATE, restore(FLAG_OFF)])]);
    assert.equal(both.v2.restores[0].tailAvailable, true);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("A1's request A differing from A0's is void-determinism, counted before any drop", {
  skip: !fs.existsSync(DEFAULT_LIB),
}, async () => {
  const lib = await import(pathToFileURL(DEFAULT_LIB).href);
  const ids = ["p0", "p1"];
  const mk = (arm, over = {}) => ({
    ...recMeta(arm),
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
  const mk = (arm) => ({ ...recMeta(arm), prompts: ids.map((id) => row(id)) });
  const records = ["A0", "A1", "C", "A5", "A3"].map(mk);
  const arms = ["A0", "A1", "C", "A5", "A3"];
  assert.equal(preScoreChecks(records, gateCensus(arms), { shell: "C" }), null);
  // benign arms launched without their perturbation: identical to A0 -> mislaunched (VOID)
  const noA5 = { ...gateCensus(arms), A5: rawCensus([F_OFF, ...ARMED]) };
  const r = preScoreChecks(records, noA5, { shell: "C" });
  assert.equal(r.verdict, "mislaunched");
  assert.ok(/extra=''/.test(r.voidReason));
  const wrongA3 = { ...gateCensus(arms), A3: rawCensus([flags(1, 0, 0, "-fa 1"), ...ARMED]) };
  assert.equal(preScoreChecks(records, wrongA3, { shell: "C" }).verdict, "mislaunched");
  // C launched with -fa 0 is mislaunched too
  const cFa0 = { ...gateCensus(arms), C: rawCensus([flags(1, 1, 0, "-fa 0"), ...ARMED]) };
  assert.equal(preScoreChecks(records, cFa0, { shell: "C" }).verdict, "mislaunched");
  // whitespace in the recorded extra does not matter
  const spaced = {
    ...gateCensus(arms),
    A5: rawCensus([flags(1, 0, 0, "  -no-fmoe   -no-fug "), ...ARMED]),
  };
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
