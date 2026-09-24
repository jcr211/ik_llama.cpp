#!/usr/bin/env node
// SV2-E1 read-only divergence census over llama-server stderr logs.
//
// Parses two line families:
//   legacy (every build):  "======== Cache: cache_size = N, n_past =  P, n_past_prompt = Q",
//                          "restored context checkpoint took X ms (pos_min = a, pos_max = b, ...",
//                          "forcing full prompt re-processing ..." / "... reprocessing from scratch",
//                          "prompt eval time = X ms / N tokens" (one per timed request)
//   State-OS v2 (LONGSPEAR_STATEOS_DIV_LOG=1): "[stateos-div] event=... key=value ..." and
//                          "[ckpt-xcheck] ..." (LONGSPEAR_STATEOS_TAIL_XCHECK=1)
//
// Tail distance = cached tokens - first divergent cache index (1 = only the last cached token
// differs). Gap = common tokens re-prefilled after the restore: n_past - (restored pos_max + 1).
//
// Usage: node tools/stateos-div-census.mjs [--json] [log ...]
//        (default log: D:/AI/ik_llama-qwen4exp/ik-serve-8099.err.log)
// Reads only; prints a text report (or one JSON object with --json) to stdout.

import fs from "node:fs";
import readline from "node:readline";
import { fileURLToPath } from "node:url";
import path from "node:path";

export const DEFAULT_LOG = "D:/AI/ik_llama-qwen4exp/ik-serve-8099.err.log";
export const BUCKETS = ["1", "2-5", "6-64", "65-512", ">512"];

export function bucketOf(tailDist) {
  if (tailDist <= 1) return "1";
  if (tailDist <= 5) return "2-5";
  if (tailDist <= 64) return "6-64";
  if (tailDist <= 512) return "65-512";
  return ">512";
}

const RE_CACHE = /======== Cache: cache_size = (\d+), n_past =\s+(\d+), n_past_prompt = (\d+)/;
const RE_RESTORED =
  /restored context checkpoint took\s+([\d.]+) ms \(pos_min = (-?\d+), pos_max = (-?\d+)/;
const RE_FORCED =
  /forcing full prompt re-processing|no checkpoint before divergence point - reprocessing from scratch/;
const RE_PROMPT_EVAL = /prompt eval time =\s+([\d.]+) ms \/\s+(\d+) tokens/;
const RE_EVAL = /^\s+eval time =\s+([\d.]+) ms \/\s+(\d+) tokens/;
const RE_VERIFY_FAIL = /restore position mismatch/;
// bracketed token windows may hold quoted pieces containing ']' or spaces
const RE_KV = /([A-Za-z_][A-Za-z0-9_]*)=("(?:[^"\\]|\\.)*"|\[(?:"(?:[^"\\]|\\.)*"|[^\]"])*\]|\S+)/g;

/** key=value fields of a [stateos-div] / [ckpt-xcheck] line; bracketed windows kept verbatim. */
export function parseFields(line) {
  const out = {};
  const start = line.indexOf("]");
  const body = start >= 0 ? line.slice(start + 1) : line;
  for (const m of body.matchAll(RE_KV)) out[m[1]] = m[2];
  return out;
}

const num = (v) => (v === undefined ? null : Number(v));

// ---- launch flags (mislaunch is decided from these, never from the lever's own output) ----------

export const FLAG_NAMES = [
  "LONGSPEAR_STATEOS_DIV_LOG",
  "LONGSPEAR_STATEOS_TAIL_SNAPSHOT",
  "LONGSPEAR_STATEOS_TAIL_XCHECK",
  "LONGSPEAR_PLE_HIST_REWIND",
  "LONGSPEAR_PLE_HIST_LOG",
];

// extra = the launcher's -ExtraArgs, whitespace-normalized ('' for every run but the benign gate arms);
// spec = the launcher's speculation state ('on', or 'off' with -SpecOff); null = either, checked
// elsewhere (step 4: every arm must share one state, see tools/stateos-tail-gate.mjs)
const flagSet = (div, tail, xcheck, extra = "", spec = "on") => ({
  LONGSPEAR_STATEOS_DIV_LOG: div,
  LONGSPEAR_STATEOS_TAIL_SNAPSHOT: tail,
  LONGSPEAR_STATEOS_TAIL_XCHECK: xcheck,
  LONGSPEAR_PLE_HIST_REWIND: "1",
  LONGSPEAR_PLE_HIST_LOG: "1",
  extra,
  spec,
});

/** The flags each W-SV2 run must have been launched with (merged plan section 4). */
export const REQUIRED_FLAGS = {
  step1: flagSet("1", "0", "0"), // telemetry read: DIV_LOG only
  step2: flagSet("1", "1", "1"), // mechanism probe: TAIL_SNAPSHOT + TAIL_XCHECK + DIV_LOG
  P0: flagSet("1", "0", "0"), // step 3 flag-off arm
  T1: flagSet("1", "1", "0"), // step 3 tail arm
  gateOff: flagSet("1", "0", "0", "", null), // step 4 A0, A1
  gateA5: flagSet("1", "0", "0", "-no-fmoe -no-fug", null), // step 4 benign A5
  gateA3: flagSet("1", "0", "0", "-fa 0", null), // step 4 benign A3
  gateTail: flagSet("1", "1", "0", "", null), // step 4 C
};

export const normalizeExtra = (s) =>
  String(s ?? "")
    .trim()
    .replace(/\s+/g, " ");

/**
 * null when the recorded flags equal the required set (env flags AND -ExtraArgs), else why not.
 * problem: a record-level problem from feedFiles (flags unknown for one log, or inconsistent records).
 */
export function flagsMismatch(flags, required, problem = null) {
  if (problem) return problem;
  if (!flags)
    return "flags unknown (no [stateos-flags] record: no <LogStem>.flags sidecar and no header line)";
  const bad = FLAG_NAMES.filter((k) => flags[k] !== required[k]).map(
    (k) => `${k}=${flags[k]} (need ${required[k]})`,
  );
  if (normalizeExtra(flags.extra) !== normalizeExtra(required.extra)) {
    bad.push(`extra='${flags.extra}' (need '${required.extra}')`);
  }
  if (required.spec != null && flags.spec !== required.spec) {
    bad.push(`spec=${flags.spec} (need ${required.spec})`);
  }
  return bad.length ? `wrong flags: ${bad.join(", ")}` : null;
}

/** <LogStem>.port: written by check-stateos-port-8099.ps1 after the launch (the ONLY port evidence). */
export function portSidecarOf(logPath) {
  return logPath.replace(/\.err\.log$/i, "").replace(/\.log$/i, "") + ".port";
}

/** <LogStem>.pid: the launched server's PID, written by launch-stateos-tail-8099.ps1. */
export function pidSidecarOf(logPath) {
  return logPath.replace(/\.err\.log$/i, "").replace(/\.log$/i, "") + ".pid";
}

// the port every W-SV2 server is launched on (launch-stateos-tail-8099.ps1 --port)
export const MODEL_PORT = 8099;

/**
 * Validate a <LogStem>.port record against <LogStem>.pid. Returns null when the record says ok, names
 * the model port, lists exactly one listener PID and that PID is the launched one; else the problem.
 * portText / pidText: the files' contents (null = file missing).
 */
export function portRecordProblem(portText, pidText, port = MODEL_PORT) {
  if (portText == null)
    return "port unverified (no <LogStem>.port record: run check-stateos-port-8099.ps1 after the launch)";
  if (pidText == null) return "port unverified (no <LogStem>.pid record of the launched PID)";
  const lines = portText.split(/\r?\n/).filter((l) => l.trim() !== "");
  const m =
    lines.length === 1 &&
    /^\[stateos-port\] (\S+) pid=(\d+) listeners=([\d,]*) port=(\d+)\s*$/.exec(lines[0]);
  if (!m) return `malformed port record: ${JSON.stringify(portText.trim().slice(0, 200))}`;
  const [, state, pid, listenerList, recPort] = m;
  const launched = pidText.trim();
  const listeners = listenerList.split(",").filter(Boolean);
  const bad = [];
  if (state !== "ok") bad.push(`state=${state}`);
  if (Number(recPort) !== port) bad.push(`port=${recPort} (need ${port})`);
  if (!/^\d+$/.test(launched) || pid !== launched) bad.push(`pid=${pid} (launched ${launched})`);
  if (listeners.length !== 1 || listeners[0] !== launched)
    bad.push(`listeners=${listenerList} (need exactly ${launched})`);
  return bad.length
    ? `mislaunched: port shared or not owned by the launched PID (${bad.join(", ")})`
    : null;
}

export function flagsSidecarOf(logPath) {
  return logPath.replace(/\.err\.log$/i, "").replace(/\.log$/i, "") + ".flags";
}

/**
 * Token ids of a cache_win / prompt_win field (`[id:"piece" *id:"piece" ...]`, '*' marks the token at
 * the divergence): {ids, center} with center = index of the marked token in ids, -1 when absent.
 */
export function parseWindow(field) {
  const out = { ids: [], center: -1 };
  if (typeof field !== "string") return out;
  for (const m of field.matchAll(/(\*?)(-?\d+):"(?:[^"\\]|\\.)*"/g)) {
    if (m[1] === "*") out.center = out.ids.length;
    out.ids.push(Number(m[2]));
  }
  return out;
}

function sum(xs) {
  return xs.reduce((a, b) => a + b, 0);
}

function pct(xs, q) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(s.length * q))];
}

export function newCensus() {
  return {
    legacy: {
      timedRequests: 0,
      promptTokens: 0,
      promptMs: 0,
      evalTokens: 0,
      evalMs: 0,
      restores: [],
      forced: [],
      verifyFailures: 0,
      restoresWithoutCacheLine: 0,
    },
    v2: {
      restores: [],
      creates: {},
      tailSkips: {},
      tailShaMismatch: 0,
      tailEvents: 0,
      xcheck: [],
      xcheckSkips: {},
    },
    ple: { sets: 0, setsBySite: {}, resetsAtPos0: 0, resetsAfterPos0: 0 },
    // the launcher's recorded env flags ([stateos-flags] line, from the <LogStem>.flags sidecar or a
    // header line in the log); null = flags unknown
    flags: null,
    flagsConflict: false, // two different records for one log (e.g. sidecar vs header line)
    flagsProblem: null, // set by feedFiles: a log without a record, or logs with different records
    // server start: "HTTP server listening" (stdout, <LogStem>.out.log) / "couldn't bind" (stderr)
    serverListening: false,
    serverBindFailed: false,
    serversAll: null, // set by feedFiles: every log's server served
    // port ownership, written by check-stateos-port-8099.ps1 into <LogStem>.port after the launch:
    // exactly one listener on the model port and it is the launched PID; null = not verified
    port: null,
    portProblem: null, // set by feedFiles: a log without a port record, or a record that is not ok
    cudaErrors: 0,
    logStem: null, // the flags record's logstem= (one log); feedFiles: null unless exactly one log
    _last: null,
    _tailPending: new Map(),
  };
}

/** Feed one log line. Pure apart from mutating `c`. */
export function feed(c, line) {
  if (line.startsWith("[stateos-flags]")) {
    const f = parseFields(line);
    const flags = {};
    for (const k of FLAG_NAMES) flags[k] = f[k] === "1" ? "1" : "0";
    // extra='...' may hold spaces: its own pattern, not parseFields
    const m = /extra='([^']*)'/.exec(line);
    flags.extra = normalizeExtra(m ? m[1] : "");
    // spec=on|off (launcher -SpecOff); a record without it is "unknown", never assumed on
    const s = / spec=(on|off)\b/.exec(line);
    flags.spec = s ? s[1] : "unknown";
    // the launcher's -LogStem: ties an arm record to its own server log (not part of the flag set, so
    // two logs of one census can agree on flags while naming different stems)
    const ls = / logstem=(\S+)/.exec(line);
    c.logStem = ls ? ls[1] : null;
    if (c.flags && JSON.stringify(c.flags) !== JSON.stringify(flags)) c.flagsConflict = true;
    c.flags = flags;
    return;
  }
  // [stateos-port] lines are NOT read from logs: port evidence comes only from the <LogStem>.port
  // sidecar, validated against <LogStem>.pid (feedFiles -> portRecordProblem)
  if (line.startsWith("[stateos-port]")) return;
  if (/HTTP server listening/.test(line)) {
    c.serverListening = true;
    return;
  }
  if (/couldn't bind to server socket/.test(line)) {
    c.serverBindFailed = true;
    return;
  }
  if (line.startsWith("[stateos-div]")) {
    const f = parseFields(line);
    if (f.event === "restore") {
      const slot = f.slot ?? "0";
      const r = {
        cacheN: num(f.cache_n),
        nPast: num(f.n_past),
        nPastPrompt: num(f.n_past_prompt),
        tailDist: num(f.tail_dist),
        bucket: f.bucket,
        cls: f.class,
        prevStop: f.prev_stop,
        prevRound: f.prev_round,
        prevNAcc: num(f.prev_n_acc),
        chosenOrigin: f.chosen_origin,
        gap: num(f.gap),
        restoreMs: num(f.restore_ms),
        reason: f.reason,
        outcome: f.outcome,
        tailAvailable: c._tailPending.get(slot) === true,
        cacheWin: parseWindow(f.cache_win),
        promptWin: parseWindow(f.prompt_win),
      };
      c._tailPending.set(slot, false);
      c.v2.restores.push(r);
    } else if (f.event === "create") {
      c.v2.creates[f.origin] = (c.v2.creates[f.origin] ?? 0) + 1;
      if (f.origin === "tail") c._tailPending.set(f.slot ?? "0", true);
    } else if (f.event === "tail_skip") {
      // a newer release produced no tail: an older pending tail no longer describes this slot
      c.v2.tailSkips[f.cause] = (c.v2.tailSkips[f.cause] ?? 0) + 1;
      c._tailPending.set(f.slot ?? "0", false);
    } else if (f.event === "tail_sha_mismatch") {
      c.v2.tailShaMismatch += 1;
    }
    if (f.event && f.event.startsWith("tail")) c.v2.tailEvents += 1;
    return;
  }
  if (line.startsWith("[ple-hist]")) {
    // LONGSPEAR_PLE_HIST_LOG=1: "set ... site=X" at every rewind site, "reset seq= pos= next_pos="
    // when the input builder had to restart the history at pos > 0 (a rewind nothing repaired)
    const f = parseFields(line);
    if (/^\[ple-hist\] set /.test(line)) {
      c.ple.sets += 1;
      c.ple.setsBySite[f.site ?? "?"] = (c.ple.setsBySite[f.site ?? "?"] ?? 0) + 1;
    } else if (/^\[ple-hist\] reset /.test(line)) {
      if (num(f.pos) > 0) c.ple.resetsAfterPos0 += 1;
      else c.ple.resetsAtPos0 += 1;
    }
    return;
  }
  if (/CUDA error/.test(line)) {
    c.cudaErrors += 1;
  }
  if (line.startsWith("[ckpt-xcheck]")) {
    const f = parseFields(line);
    if (f.skip !== undefined) {
      c.v2.xcheckSkips[f.skip] = (c.v2.xcheckSkips[f.skip] ?? 0) + 1;
    } else if (f.layer !== undefined) {
      c.v2.xcheck.push({
        layer: num(f.layer),
        nBitequal: num(f.n_bitequal),
        n: num(f.n),
        relL2: num(f.relL2),
      });
    }
    return;
  }
  let m = RE_CACHE.exec(line);
  if (m) {
    c._last = { size: +m[1], nPast: +m[2], nPastPrompt: +m[3] };
    return;
  }
  m = RE_RESTORED.exec(line);
  if (m) {
    if (!c._last) {
      c.legacy.restoresWithoutCacheLine += 1;
      return;
    }
    const tail = c._last.size - c._last.nPast;
    c.legacy.restores.push({ tail, gap: c._last.nPast - (+m[3] + 1), ms: +m[1], posMax: +m[3] });
    return;
  }
  if (RE_FORCED.test(line)) {
    c.legacy.forced.push({ prefix: c._last ? c._last.nPast : null });
    return;
  }
  if (RE_VERIFY_FAIL.test(line)) {
    c.legacy.verifyFailures += 1;
    return;
  }
  m = RE_PROMPT_EVAL.exec(line);
  if (m) {
    c.legacy.timedRequests += 1;
    c.legacy.promptMs += +m[1];
    c.legacy.promptTokens += +m[2];
    return;
  }
  m = RE_EVAL.exec(line);
  if (m) {
    c.legacy.evalMs += +m[1];
    c.legacy.evalTokens += +m[2];
  }
}

export function summarize(c) {
  const L = c.legacy;
  const byBucket = {};
  for (const b of BUCKETS) byBucket[b] = { n: 0, gapTokens: 0 };
  for (const r of L.restores) {
    const b = byBucket[bucketOf(r.tail)];
    b.n += 1;
    b.gapTokens += r.gap;
  }
  const gaps = L.restores.map((r) => r.gap);
  const forcedPrefixes = L.forced.map((f) => f.prefix).filter((p) => p !== null);
  const legacy = {
    timedRequests: L.timedRequests,
    promptTokens: L.promptTokens,
    prefillTokPerSec: L.promptMs > 0 ? L.promptTokens / (L.promptMs / 1000) : null,
    decodeTokPerSec: L.evalMs > 0 ? L.evalTokens / (L.evalMs / 1000) : null,
    busySeconds: (L.promptMs + L.evalMs) / 1000,
    restores: L.restores.length,
    restoresWithoutCacheLine: L.restoresWithoutCacheLine,
    gapTokens: sum(gaps),
    gapP50: pct(gaps, 0.5),
    gapP90: pct(gaps, 0.9),
    gapMax: gaps.length ? Math.max(...gaps) : null,
    byBucket,
    forced: L.forced.length,
    forcedWithPrefix: forcedPrefixes.length,
    forcedPrefixLt64: forcedPrefixes.filter((p) => p < 64).length,
    forcedPrefixMax: forcedPrefixes.length ? Math.max(...forcedPrefixes) : null,
    forcedPrefixTokens: sum(forcedPrefixes),
    verifyFailures: L.verifyFailures,
  };

  const V = c.v2;
  const rs = V.restores;
  // new-conversation resets (no checkpoint and a common prefix < 64: a different conversation, the
  // census's 491 "forced" class) are not divergences of the cached conversation
  const isNewConversation = (r) => r.outcome === "reset:no-checkpoint" && r.nPast < 64;
  const divergences = rs.filter((r) => !isNewConversation(r));
  const last = rs.filter((r) => r.tailDist === 1);
  const lastAfterDrafted = last.filter((r) => r.prevRound === "drafted" && r.prevNAcc >= 1);
  const eligibleLast = last.filter((r) => r.tailAvailable);
  // a tail restore = the search chose the tail and its restore succeeded; under the crosscheck the
  // server then continues on the flag-off checkpoint (outcome restored:xcheck-flag-off), or resets
  // when the flag-off path has no checkpoint (reset:xcheck-flag-off): both are hits
  const isTailHit = (r) =>
    r.chosenOrigin === "tail" &&
    (String(r.outcome).startsWith("restored") || r.outcome === "reset:xcheck-flag-off");
  const tailRestoredOfEligible = eligibleLast.filter(isTailHit);
  // tails that were eligible at release but not written (writer refusal, order, short cache, size)
  const WRITER_SKIPS = ["refused", "order", "cache-short", "size-mismatch"];
  const writerSkips = sum(WRITER_SKIPS.map((k) => V.tailSkips[k] ?? 0));
  const TAIL_FAILURES = ["reset:restore-failed", "reset:verify-failed", "reset:rewind-refused"];
  // step 3's comparison class, ONE definition for both runs, a property of the TRAFFIC not the lever:
  // last-token divergences whose previous generation ended with a drafted round that accepted >= 1
  // draft (shadow_pos = root - 1 <= last cached - 2 = root + n_acc - 2 iff n_acc >= 1). Computed the
  // same way from each run's own [stateos-div] lines.
  const tailOn = (V.creates.tail ?? 0) > 0 || Object.keys(V.tailSkips).length > 0;
  const eligibleClass = lastAfterDrafted;
  const eligibleGaps = eligibleClass.map((r) => r.gap ?? 0);
  // mechanism on that class (tail-on runs): a tail was written for the event and the search chose it
  // (strict: the live state after the restore is the tail's - outcome restored, reason tail)
  const eligibleServed = eligibleClass.filter(
    (r) =>
      r.tailAvailable &&
      r.chosenOrigin === "tail" &&
      r.outcome === "restored" &&
      r.reason === "tail",
  );
  // the achievable floor: a tail restore still re-prefills the final round's accepted drafts (the
  // shadow sits at root - 1, the divergence at root + n_acc), so a served event's gap is n_acc
  const eligibleFloor = eligibleClass.length
    ? sum(eligibleClass.map((r) => r.prevNAcc ?? 0)) / eligibleClass.length
    : null;
  // the crosscheck was on in this run (diagnostic: the server continues on the flag-off state)
  const xcheckActive =
    V.xcheck.length > 0 ||
    Object.keys(V.xcheckSkips).length > 0 ||
    rs.some((r) => String(r.outcome).endsWith("xcheck-flag-off"));
  const lastDivergences = divergences.filter((r) => r.tailDist === 1);
  const v2ByBucket = {};
  for (const b of BUCKETS) v2ByBucket[b] = { n: 0, gapTokens: 0 };
  for (const r of rs) {
    const b = v2ByBucket[r.bucket ?? bucketOf(r.tailDist)];
    if (b) {
      b.n += 1;
      b.gapTokens += r.gap ?? 0;
    }
  }
  const count = (xs, key) => xs.reduce((m, x) => ((m[x[key]] = (m[x[key]] ?? 0) + 1), m), {});
  const relL2 = V.xcheck.map((x) => x.relL2).filter(Number.isFinite);
  const v2 = {
    // every restore-branch decision, including new-conversation resets
    restoreDecisions: rs.length,
    // the step-1 denominator: decisions minus new-conversation resets
    divergenceEvents: divergences.length,
    newConversationResets: rs.length - divergences.length,
    tailOn,
    xcheckActive,
    byBucket: v2ByBucket,
    byClass: count(rs, "cls"),
    byOutcome: count(rs, "outcome"),
    byReason: count(rs, "reason"),
    lastToken: {
      n: last.length,
      shareOfEvents: divergences.length ? lastDivergences.length / divergences.length : null,
      afterDraftedRoundWithAccepted: lastAfterDrafted.length,
      shareAfterDraftedRoundWithAccepted: last.length
        ? lastAfterDrafted.length / last.length
        : null,
      // all last-token events: REPORT-ONLY in step 3
      gapTokens: sum(last.map((r) => r.gap ?? 0)),
      byPrevStop: count(last, "prevStop"),
      byPrevRound: count(last, "prevRound"),
      withTailAvailable: eligibleLast.length,
      writerSkips,
      restoredFromTail: tailRestoredOfEligible.length,
      // eligible-at-release tails the writer did not produce count as misses
      tailHitRate:
        eligibleLast.length + writerSkips
          ? tailRestoredOfEligible.length / (eligibleLast.length + writerSkips)
          : null,
      // step 3's class (see above): count, gap tokens, gap per event
      eligible: {
        definition: "last-token, prev_round=drafted, prev_n_acc>=1 (same in every run)",
        n: eligibleClass.length,
        gapTokens: sum(eligibleGaps),
        gapPerEvent: eligibleClass.length ? sum(eligibleGaps) / eligibleClass.length : null,
        // tail written for the event AND chosen by the search (meaningful for tail-on runs)
        served: eligibleServed.length,
        servedRate: eligibleClass.length ? eligibleServed.length / eligibleClass.length : null,
        // mean prev_n_acc over the class: the gap a served event still pays
        floor: eligibleFloor,
      },
    },
    creates: V.creates,
    tailSkips: V.tailSkips,
    tailShaMismatch: V.tailShaMismatch,
    tailRestores: rs.filter(isTailHit).length,
    verifyFailedAfterTailChoice: rs.filter(
      (r) => r.chosenOrigin === "tail" && r.outcome === "reset:verify-failed",
    ).length,
    // restore-failed / verify-failed / rewind-refused after the search chose a tail
    failedAfterTailChoice: rs.filter(
      (r) => r.chosenOrigin === "tail" && TAIL_FAILURES.includes(r.outcome),
    ).length,
    restoreFailedAfterTailChoice: rs.filter((r) => r.chosenOrigin === "tail" && !isTailHit(r))
      .length,
    xcheck: {
      rows: V.xcheck.length,
      skips: V.xcheckSkips,
      relL2Max: relL2.length ? Math.max(...relL2) : null,
      relL2P50: pct(relL2, 0.5),
      bitEqualRows: V.xcheck.filter((x) => x.n !== null && x.nBitequal === x.n).length,
    },
  };
  const served = c.serversAll ?? {
    ok: c.serverListening && !c.serverBindFailed,
    detail: `listening=${c.serverListening} bind-failed=${c.serverBindFailed}`,
  };
  return {
    legacy,
    v2,
    ple: c.ple,
    cudaErrors: c.cudaErrors,
    flags: c.flags,
    flagsProblem:
      c.flagsProblem ??
      (c.flagsConflict ? "flags inconsistent (two different records for one log)" : null),
    served,
    portProblem: portProblemOf(c),
  };
}

/** null when the port record says the launched PID was the only listener, else why not. */
export function portProblemOf(c) {
  if (c.portProblem) return c.portProblem;
  if (!c.port) return portRecordProblem(null, null);
  return c.port.ok ? null : c.port.detail;
}

/** Record one log's port evidence (the <LogStem>.port / .pid contents, null = missing) in `c`. */
export function setPortRecord(c, portText, pidText) {
  const problem = portRecordProblem(portText, pidText);
  c.port = { ok: problem === null, detail: problem ?? portText.trim() };
}

const miss = (name, ok, detail) => ({ name, ok: Boolean(ok), detail });

// A check that fails with `void: true` makes the step VOID (the measurement cannot answer: traffic
// mismatch or a protocol error), not a lever kill. Any other failed check is a MISS (stop).
const voidCheck = (name, ok, detail) => ({ name, ok: Boolean(ok), detail, void: true });

// step-3 rules (coordinator rulings on Opus S2 / N3 and the Grok re-review; stated in .lane/PROGRESS.md
// and the gpu-justify file):
// - ONE eligibility class in both runs, from the traffic: last-token divergences whose previous
//   generation ended with a drafted round that accepted >= 1 draft;
// - traffic sanity (VOID when violated): each run has >= STEP3_MIN_EVENTS eligible events and T1's
//   eligible count is within [STEP3_COUNT_LO, STEP3_COUNT_HI] x P0's;
// - mechanism (MISS): in T1 a tail was written and chosen on >= STEP3_MIN_SERVED of the eligible events;
// - effect (MISS): >= 90 % of the ACHIEVABLE reduction: T1 mean gap <= floor_T1 + 0.10 x (P0 mean gap
//   - floor_P0), floor = mean prev_n_acc of the class in that run (a tail-served event's gap is n_acc);
//   VOID when P0's mean gap <= 1.5 x floor_P0 (gap too small to measure);
// - protocol (VOID, "mislaunched"): the launcher's RECORDED flags (<LogStem>.flags) differ from
//   REQUIRED_FLAGS.P0 / .T1, or are missing ("flags unknown"). Never inferred from the lever's output.
// The raw per-event ratio and the all-last-token-events ratio are report-only.
export const STEP3_FLOOR_MARGIN = 1.5;
export const STEP3_REDUCTION_SLACK = 0.1;
export const STEP3_MIN_EVENTS = 5;
export const STEP3_COUNT_LO = 0.5;
export const STEP3_COUNT_HI = 2;
export const STEP3_MIN_SERVED = 0.9;

// writer outcomes that must never happen for an eligible tail (step 2)
const WRITER_FAILURE_SKIPS = ["refused", "order", "cache-short", "size-mismatch"];

/**
 * W-SV2 mechanism checks (merged plan section 4) on one step's summary; step 3 compares P0 with T1.
 * Every step also requires the PLE history repair to be armed and complete: at least one
 * "[ple-hist] set" line (LONGSPEAR_PLE_HIST_REWIND=1 + LONGSPEAR_PLE_HIST_LOG=1) and zero
 * "[ple-hist] reset" at pos > 0, and zero CUDA error lines.
 * Returns {verdict: PASS | STOP | VOID, pass, checks}.
 */
export function checkStep(step, s, p0 = null) {
  const V = s.v2;
  const T = V.lastToken;
  const checks = [
    miss("ple-hist armed", s.ple.sets > 0, `[ple-hist] set lines=${s.ple.sets}`),
    miss(
      "ple-hist reset at pos > 0 == 0",
      s.ple.resetsAfterPos0 === 0,
      `resets=${s.ple.resetsAfterPos0}`,
    ),
    // a CUDA error is always a STOP, even when a traffic/protocol check would VOID the step
    { ...miss("CUDA errors == 0", s.cudaErrors === 0, `lines=${s.cudaErrors}`), hard: true },
  ];
  // steps 2 and 3: a tail choice never ends in a failed restore, and no tail prefix mismatches. These
  // are C1 correctness failures on any traffic: hard, they STOP even when a VOID check also fails.
  const tailIntegrity = () => {
    checks.push({
      ...miss(
        "restore-failed/verify-failed/rewind-refused after a tail choice == 0",
        V.failedAfterTailChoice === 0,
        `n=${V.failedAfterTailChoice}`,
      ),
      hard: true,
    });
    checks.push({
      ...miss("sha mismatches == 0", V.tailShaMismatch === 0, `n=${V.tailShaMismatch}`),
      hard: true,
    });
  };
  // mislaunch (VOID) is decided ONLY from the launcher's recorded flags, never from the lever's output
  const launched = (who, run, required) => {
    const why = flagsMismatch(run.flags, required, run.flagsProblem);
    checks.push(voidCheck(`mislaunched? ${who} flags`, why === null, why ?? "flags as required"));
    // the chain's port check: the launched PID was the only listener (SO_REUSEADDR lets a second
    // server bind on Windows, so the log's bind-failure line cannot be trusted for this)
    const port = run.portProblem ?? null;
    checks.push(
      voidCheck(
        `mislaunched? ${who} port`,
        port === null,
        port ?? "launched PID is the only listener",
      ),
    );
  };
  // steps 1 and 2: the log must show that this server served (build lines, not lever output):
  // "HTTP server listening" present, no "couldn't bind" (another server held the port)
  const servedCheck = () =>
    checks.push(
      voidCheck(
        "server served (listening, no bind failure)",
        s.served?.ok === true,
        s.served?.detail ?? "unknown",
      ),
    );
  if (step === "step1") {
    launched("step-1 run", s, REQUIRED_FLAGS.step1);
    servedCheck();
    checks.push(
      miss(
        "divergence events > 0 (new-conversation resets excluded)",
        V.divergenceEvents > 0,
        `events=${V.divergenceEvents} (decisions=${V.restoreDecisions}, new-conversation resets=${V.newConversationResets})`,
      ),
    );
    checks.push(
      miss("last-token share >= 0.20", (T.shareOfEvents ?? 0) >= 0.2, `share=${T.shareOfEvents}`),
    );
    checks.push(
      miss(
        "last-token after a drafted round with >= 1 accepted >= 0.50",
        (T.shareAfterDraftedRoundWithAccepted ?? 0) >= 0.5,
        `share=${T.shareAfterDraftedRoundWithAccepted}`,
      ),
    );
  } else if (step === "step2") {
    // the probe is TAIL_SNAPSHOT=1 TAIL_XCHECK=1 DIV_LOG=1 (launch-stateos-tail-xcheck-8099.ps1)
    launched("step-2 probe", s, REQUIRED_FLAGS.step2);
    servedCheck();
    checks.push(
      miss(
        "tail available on >= 1 last-token divergence",
        T.withTailAvailable > 0,
        `n=${T.withTailAvailable}`,
      ),
    );
    checks.push(
      miss(
        "origin=tail restores >= 0.90 of eligible (writer skips count as misses)",
        (T.tailHitRate ?? 0) >= 0.9,
        `rate=${T.tailHitRate} hits=${T.restoredFromTail} available=${T.withTailAvailable} writer-skips=${T.writerSkips}`,
      ),
    );
    for (const cause of WRITER_FAILURE_SKIPS) {
      checks.push(
        miss(
          `tail_skip cause=${cause} == 0`,
          (V.tailSkips[cause] ?? 0) === 0,
          JSON.stringify(V.tailSkips),
        ),
      );
    }
    checks.push(
      miss(
        "verify failures == 0",
        s.legacy.verifyFailures === 0 && (V.byOutcome["reset:verify-failed"] ?? 0) === 0,
        `legacy=${s.legacy.verifyFailures} v2=${V.byOutcome["reset:verify-failed"] ?? 0}`,
      ),
    );
    tailIntegrity();
  } else if (step === "step3") {
    if (!p0) throw new Error("step3 needs the P0 summary");
    const P = p0.v2;
    const pe = P.lastToken.eligible;
    const te = T.eligible;
    // protocol (recorded flags) and traffic sanity: VOID, not a lever kill
    launched("P0", p0, REQUIRED_FLAGS.P0);
    launched("T1", s, REQUIRED_FLAGS.T1);
    // step 3 reads two logs: a CUDA error in the P0 log is a STOP as well (hard, beats VOID)
    checks.push({
      ...miss("P0 CUDA errors == 0", p0.cudaErrors === 0, `lines=${p0.cudaErrors}`),
      hard: true,
    });
    checks.push(
      voidCheck(
        `P0 eligible events >= ${STEP3_MIN_EVENTS}`,
        pe.n >= STEP3_MIN_EVENTS,
        `n=${pe.n} (${pe.definition})`,
      ),
    );
    checks.push(
      voidCheck(
        `T1 eligible events >= ${STEP3_MIN_EVENTS}`,
        te.n >= STEP3_MIN_EVENTS,
        `n=${te.n} (${te.definition})`,
      ),
    );
    checks.push(
      voidCheck(
        `T1 eligible count within ${STEP3_COUNT_LO}x-${STEP3_COUNT_HI}x P0's`,
        pe.n > 0 && te.n >= STEP3_COUNT_LO * pe.n && te.n <= STEP3_COUNT_HI * pe.n,
        `P0=${pe.n} T1=${te.n}`,
      ),
    );
    checks.push(
      miss(
        "P0 ple-hist armed and clean",
        p0.ple.sets > 0 && p0.ple.resetsAfterPos0 === 0,
        `sets=${p0.ple.sets} resets=${p0.ple.resetsAfterPos0}`,
      ),
    );
    // mechanism on T1
    checks.push(
      miss(
        `T1: a tail was written and chosen on >= ${STEP3_MIN_SERVED} of eligible events`,
        (te.servedRate ?? 0) >= STEP3_MIN_SERVED,
        `served=${te.served}/${te.n} rate=${te.servedRate}`,
      ),
    );
    // effect, per event on the one class, against the ACHIEVABLE reduction (coordinator ruling): a
    // served event still re-prefills the final round's accepted drafts, so its gap is n_acc. floor =
    // mean prev_n_acc over the class, from each run's own lines. Pass: T1 mean gap <= floor_T1 +
    // 0.10 x (P0 mean gap - floor_P0), i.e. >= 90 % of the achievable reduction. VOID when P0's mean
    // gap <= 1.5 x floor_P0 (nothing meaningful to save).
    const fp = pe.floor;
    const ft = te.floor;
    checks.push(
      voidCheck(
        `gap large enough to measure: P0 mean gap > ${STEP3_FLOOR_MARGIN} x floor`,
        pe.gapPerEvent !== null && fp !== null && pe.gapPerEvent > STEP3_FLOOR_MARGIN * fp,
        `P0 mean gap=${pe.gapPerEvent} floor_P0=${fp}`,
      ),
    );
    const bound =
      pe.gapPerEvent !== null && fp !== null && ft !== null
        ? ft + STEP3_REDUCTION_SLACK * (pe.gapPerEvent - fp)
        : null;
    checks.push(
      miss(
        "eligible class: T1 mean gap <= floor_T1 + 0.10 x (P0 mean gap - floor_P0) (>= 90% of the achievable reduction)",
        bound !== null && te.gapPerEvent !== null && te.gapPerEvent <= bound,
        `P0=${pe.gapTokens}/${pe.n}=${pe.gapPerEvent} (floor ${fp}) T1=${te.gapTokens}/${te.n}=${te.gapPerEvent} (floor ${ft}) bound=${bound}`,
      ),
    );
    tailIntegrity();
    // report-only, never a miss
    checks.push({
      name: "REPORT-ONLY raw per-event ratio on the eligible class",
      ok: true,
      reportOnly: true,
      detail: `T1/P0=${pe.gapPerEvent > 0 && te.gapPerEvent !== null ? te.gapPerEvent / pe.gapPerEvent : null}`,
    });
    const allP = P.lastToken.gapTokens;
    const allT = T.gapTokens;
    checks.push({
      name: "REPORT-ONLY all last-token events gap tokens",
      ok: true,
      reportOnly: true,
      detail: `P0=${allP} T1=${allT} ratio=${allP > 0 ? allT / allP : null}`,
    });
  } else {
    throw new Error(`unknown step ${step}`);
  }
  const hardFail = checks.some((c) => c.hard && !c.ok);
  const voided = checks.some((c) => c.void && !c.ok);
  const pass = checks.every((c) => c.ok);
  return {
    step,
    verdict: hardFail ? "STOP" : voided ? "VOID" : pass ? "PASS" : "STOP",
    pass,
    checks,
  };
}

/** <LogStem>.out.log next to <LogStem>.err.log (the server's stdout: "HTTP server listening"). */
export function stdoutLogOf(logPath) {
  return /\.err\.log$/i.test(logPath) ? logPath.replace(/\.err\.log$/i, ".out.log") : null;
}

/**
 * Raw census state of log files. Per log: its <LogStem>.flags sidecar (if present) is read first, then
 * the log, then the "HTTP server listening" line of its <LogStem>.out.log. Every log must carry its own
 * flags record and all records must agree; one file's flags never stand in for another's (a missing or
 * differing record sets flagsProblem, so every step check is VOID). Every log's server must have served.
 */
export async function feedFiles(files) {
  const c = newCensus();
  const perFile = [];
  for (const f of files) {
    // per-log state: records, and the parser's slot state (a pending tail or the last Cache line of one
    // log must never describe the next log)
    c.flags = null;
    c.flagsConflict = false;
    c.serverListening = false;
    c.serverBindFailed = false;
    c.port = null;
    c.logStem = null;
    c._last = null;
    c._tailPending = new Map();
    const sidecar = flagsSidecarOf(f);
    if (fs.existsSync(sidecar)) {
      for (const line of fs.readFileSync(sidecar, "utf8").split(/\r?\n/)) feed(c, line);
    }
    // port evidence: ONLY the <LogStem>.port sidecar, validated against <LogStem>.pid
    const readOrNull = (file) => (fs.existsSync(file) ? fs.readFileSync(file, "utf8") : null);
    setPortRecord(c, readOrNull(portSidecarOf(f)), readOrNull(pidSidecarOf(f)));
    const rl = readline.createInterface({ input: fs.createReadStream(f), crlfDelay: Infinity });
    for await (const line of rl) feed(c, line);
    const out = stdoutLogOf(f);
    if (out && fs.existsSync(out)) {
      const ro = readline.createInterface({ input: fs.createReadStream(out), crlfDelay: Infinity });
      for await (const line of ro) {
        if (/HTTP server listening/.test(line)) c.serverListening = true;
      }
    }
    perFile.push({
      file: f,
      flags: c.flags,
      conflict: c.flagsConflict,
      listening: c.serverListening,
      bindFailed: c.serverBindFailed,
      port: c.port,
      logStem: c.logStem,
    });
  }
  c.logStem = perFile.length === 1 ? perFile[0].logStem : null;
  const portBad = perFile.filter((p) => !p.port || !p.port.ok);
  c.port =
    perFile.length && !portBad.length
      ? { ok: true, detail: "every log's launched PID was the only listener" }
      : null;
  c.portProblem = portBad.length
    ? portBad
        .map((p) => `${p.file}: ${p.port ? p.port.detail : portRecordProblem(null, null)}`)
        .join("; ")
    : null;
  const missing = perFile.filter((p) => !p.flags).map((p) => p.file);
  const conflicting = perFile.filter((p) => p.conflict).map((p) => p.file);
  const distinct = new Set(perFile.filter((p) => p.flags).map((p) => JSON.stringify(p.flags)));
  c.flags = perFile.length && !missing.length && distinct.size === 1 ? perFile[0].flags : null;
  c.flagsProblem = missing.length
    ? `flags unknown for ${missing.join(", ")} (no <LogStem>.flags sidecar and no header line)`
    : conflicting.length
      ? `flags inconsistent within ${conflicting.join(", ")} (sidecar and header line differ)`
      : distinct.size > 1
        ? "flags inconsistent across logs (each log is checked against its own record; they differ)"
        : null;
  const notServed = perFile.filter((p) => !p.listening || p.bindFailed);
  c.serversAll = {
    ok: perFile.length > 0 && notServed.length === 0,
    detail: notServed.length
      ? notServed
          .map((p) => `${p.file}: listening=${p.listening} bind-failed=${p.bindFailed}`)
          .join("; ")
      : "every log's server listened",
  };
  return c;
}

export async function censusOfFiles(files) {
  return summarize(await feedFiles(files));
}

function fmt(x, digits = 0) {
  return x === null || x === undefined ? "n/a" : Number(x).toFixed(digits);
}

export function renderText(files, s) {
  const L = s.legacy;
  const out = [];
  out.push(`files: ${files.join(", ")}`);
  out.push(
    `legacy: timed requests=${L.timedRequests} prefill=${fmt(L.prefillTokPerSec)} t/s decode=${fmt(L.decodeTokPerSec, 1)} t/s busy=${fmt(L.busySeconds)} s`,
  );
  out.push(
    `legacy restores=${L.restores} gap tokens=${L.gapTokens} (p50 ${fmt(L.gapP50)}, p90 ${fmt(L.gapP90)}, max ${fmt(L.gapMax)}); restores without a Cache line=${L.restoresWithoutCacheLine}; verify failures=${L.verifyFailures}`,
  );
  for (const b of BUCKETS) {
    out.push(`  tail distance ${b}: n=${L.byBucket[b].n} gap tokens=${L.byBucket[b].gapTokens}`);
  }
  out.push(
    `legacy forced full re-processing=${L.forced} (with a Cache line ${L.forcedWithPrefix}; common prefix < 64: ${L.forcedPrefixLt64}; max prefix ${fmt(L.forcedPrefixMax)}; prefix tokens ${L.forcedPrefixTokens})`,
  );
  out.push(
    `launch flags: ${s.flags ? JSON.stringify(s.flags) : "unknown (no [stateos-flags] record)"}`,
  );
  out.push(
    `ple-hist: set lines=${s.ple.sets} ${JSON.stringify(s.ple.setsBySite)}; resets at pos > 0=${s.ple.resetsAfterPos0} (at pos 0: ${s.ple.resetsAtPos0}); CUDA error lines=${s.cudaErrors}`,
  );
  const V = s.v2;
  if (V.divergenceEvents === 0 && Object.keys(V.creates).length === 0) {
    out.push("stateos-div: no [stateos-div] lines");
    return out.join("\n");
  }
  out.push(
    `stateos-div: divergence events=${V.divergenceEvents} creates=${JSON.stringify(V.creates)}`,
  );
  for (const b of BUCKETS) {
    out.push(`  tail distance ${b}: n=${V.byBucket[b].n} gap tokens=${V.byBucket[b].gapTokens}`);
  }
  out.push(`  by class ${JSON.stringify(V.byClass)}`);
  out.push(
    `  by outcome ${JSON.stringify(V.byOutcome)}; by restored origin ${JSON.stringify(V.byReason)}`,
  );
  const T = V.lastToken;
  out.push(
    `  last-token: n=${T.n} share=${fmt(T.shareOfEvents, 3)} after drafted round with >=1 accepted=${T.afterDraftedRoundWithAccepted} (share ${fmt(T.shareAfterDraftedRoundWithAccepted, 3)}) gap tokens=${T.gapTokens}`,
  );
  out.push(
    `  last-token prev stop ${JSON.stringify(T.byPrevStop)}; prev round ${JSON.stringify(T.byPrevRound)}`,
  );
  out.push(
    `  tail: available on ${T.withTailAvailable} last-token divergences, restored from tail on ${T.restoredFromTail} (hit rate ${fmt(T.tailHitRate, 3)}); skips ${JSON.stringify(V.tailSkips)}; sha mismatches ${V.tailShaMismatch}; verify failures after a tail choice ${V.verifyFailedAfterTailChoice}; non-restored tail choices ${V.restoreFailedAfterTailChoice}`,
  );
  if (V.xcheck.rows || Object.keys(V.xcheck.skips).length) {
    out.push(
      `  xcheck: rows=${V.xcheck.rows} bit-equal rows=${V.xcheck.bitEqualRows} relL2 p50=${V.xcheck.relL2P50} max=${V.xcheck.relL2Max} skips=${JSON.stringify(V.xcheck.skips)}`,
    );
  }
  return out.join("\n");
}

// Usage: [--json] [log ...] | --check step1|step2 <log> | --check step3 --p0 <P0 log> <T1 log>
// --check exits 0 on PASS, 2 on STOP (a miss: the chain script's auto-stop, the lever's kill), and
// 3 on VOID (traffic mismatch or protocol error: the measurement did not answer; not a C1 kill).
async function main(argv) {
  const json = argv.includes("--json");
  const ci = argv.indexOf("--check");
  const pi = argv.indexOf("--p0");
  const step = ci >= 0 ? argv[ci + 1] : null;
  const p0File = pi >= 0 ? argv[pi + 1] : null;
  const skip = new Set([...(ci >= 0 ? [ci, ci + 1] : []), ...(pi >= 0 ? [pi, pi + 1] : [])]);
  const files = argv.filter((a, i) => a !== "--json" && !skip.has(i));
  if (!files.length) files.push(DEFAULT_LOG);
  const s = await censusOfFiles(files);
  if (step) {
    const p0 = p0File ? await censusOfFiles([p0File]) : null;
    const r = checkStep(step, s, p0);
    for (const c of r.checks)
      process.stdout.write(
        `${c.reportOnly ? "REPORT" : c.ok ? "PASS" : c.void ? "VOID" : "MISS"} ${c.name}: ${c.detail}\n`,
      );
    process.stdout.write(
      `${step}: ${r.verdict}${r.verdict === "VOID" ? " (the measurement cannot answer; not a C1 kill)" : ""}\n`,
    );
    process.exitCode = r.verdict === "PASS" ? 0 : r.verdict === "VOID" ? 3 : 2;
    return;
  }
  if (json) {
    process.stdout.write(`${JSON.stringify({ files, ...s }, null, 2)}\n`);
  } else {
    process.stdout.write(`${renderText(files, s)}\n`);
  }
}

const isEntryPoint =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isEntryPoint) {
  main(process.argv.slice(2)).catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
