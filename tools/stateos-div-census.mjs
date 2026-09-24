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
const RE_KV = /([A-Za-z_][A-Za-z0-9_]*)=("(?:[^"\\]|\\.)*"|\[[^\]]*\]|\S+)/g;

/** key=value fields of a [stateos-div] / [ckpt-xcheck] line; bracketed windows kept verbatim. */
export function parseFields(line) {
  const out = {};
  const start = line.indexOf("]");
  const body = start >= 0 ? line.slice(start + 1) : line;
  for (const m of body.matchAll(RE_KV)) out[m[1]] = m[2];
  return out;
}

const num = (v) => (v === undefined ? null : Number(v));

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
    _last: null,
    _tailPending: new Map(),
  };
}

/** Feed one log line. Pure apart from mutating `c`. */
export function feed(c, line) {
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
      };
      c._tailPending.set(slot, false);
      c.v2.restores.push(r);
    } else if (f.event === "create") {
      c.v2.creates[f.origin] = (c.v2.creates[f.origin] ?? 0) + 1;
      if (f.origin === "tail") c._tailPending.set(f.slot ?? "0", true);
    } else if (f.event === "tail_skip") {
      c.v2.tailSkips[f.cause] = (c.v2.tailSkips[f.cause] ?? 0) + 1;
    } else if (f.event === "tail_sha_mismatch") {
      c.v2.tailShaMismatch += 1;
    }
    if (f.event && f.event.startsWith("tail")) c.v2.tailEvents += 1;
    return;
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
  const last = rs.filter((r) => r.tailDist === 1);
  const lastAfterDrafted = last.filter((r) => r.prevRound === "drafted" && r.prevNAcc >= 1);
  const eligibleLast = last.filter((r) => r.tailAvailable);
  // a tail restore = the search chose the tail and its restore succeeded; under the crosscheck the
  // server then continues on the flag-off checkpoint (outcome restored:xcheck-flag-off)
  const tailRestoredOfEligible = eligibleLast.filter(
    (r) => r.chosenOrigin === "tail" && String(r.outcome).startsWith("restored"),
  );
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
    divergenceEvents: rs.length,
    byBucket: v2ByBucket,
    byClass: count(rs, "cls"),
    byOutcome: count(rs, "outcome"),
    byReason: count(rs, "reason"),
    lastToken: {
      n: last.length,
      shareOfEvents: rs.length ? last.length / rs.length : null,
      afterDraftedRoundWithAccepted: lastAfterDrafted.length,
      shareAfterDraftedRoundWithAccepted: last.length
        ? lastAfterDrafted.length / last.length
        : null,
      gapTokens: sum(last.map((r) => r.gap ?? 0)),
      byPrevStop: count(last, "prevStop"),
      byPrevRound: count(last, "prevRound"),
      withTailAvailable: eligibleLast.length,
      restoredFromTail: tailRestoredOfEligible.length,
      tailHitRate: eligibleLast.length ? tailRestoredOfEligible.length / eligibleLast.length : null,
    },
    creates: V.creates,
    tailSkips: V.tailSkips,
    tailShaMismatch: V.tailShaMismatch,
    verifyFailedAfterTailChoice: rs.filter(
      (r) => r.chosenOrigin === "tail" && r.outcome === "reset:verify-failed",
    ).length,
    restoreFailedAfterTailChoice: rs.filter(
      (r) => r.chosenOrigin === "tail" && !String(r.outcome).startsWith("restored"),
    ).length,
    xcheck: {
      rows: V.xcheck.length,
      skips: V.xcheckSkips,
      relL2Max: relL2.length ? Math.max(...relL2) : null,
      relL2P50: pct(relL2, 0.5),
      bitEqualRows: V.xcheck.filter((x) => x.n !== null && x.nBitequal === x.n).length,
    },
  };
  return { legacy, v2 };
}

export async function censusOfFiles(files) {
  const c = newCensus();
  for (const f of files) {
    const rl = readline.createInterface({ input: fs.createReadStream(f), crlfDelay: Infinity });
    for await (const line of rl) feed(c, line);
  }
  return summarize(c);
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

async function main(argv) {
  const json = argv.includes("--json");
  const files = argv.filter((a) => a !== "--json");
  if (!files.length) files.push(DEFAULT_LOG);
  const s = await censusOfFiles(files);
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
