#!/usr/bin/env node
// SV2-E1 W-SV2 step 4 driver: the standing v2 fidelity gate (docs/NOISE-FLOOR-V2-20260908.md,
// preregistered in docs/drafts/parity-noise-floor-20260906.md v2) applied to C1, the tail snapshot.
// COORDINATOR-RUN. It talks to an already-running llama-server; it never launches one.
//
// Per prompt (the 24 v2 prompts, read from the v2 campaign receipt and sha-checked):
//   1. erase slot 0 (best effort), tokenize the prompt;
//   2. request A: greedy, n_predict = --n-first, cache_prompt: true -> generated ids g[0..G-1].
//      After A the slot caches prompt + g[0..G-2] (the last sampled token is never decoded);
//   3. request B: prompt + g[0..G-3] + [X], X != g[G-2]: the prompt diverges at the LAST cached token
//      (tail distance 1, the class C1 targets); greedy continuation of --horizon tokens.
//   The arm's record is B's continuation ids. The server log's [stateos-div] lines (census tool)
//   confirm the forced divergence landed at tail distance 1 and which checkpoint served it.
//
// Arms (one server launch each, speculation config identical, flags per arm):
//   A0 flag off, A1 flag-off repeat (fresh process, determinism control), C flag on
//   (LONGSPEAR_STATEOS_TAIL_SNAPSHOT=1), benign A5 (-no-fmoe -no-fug) and A3 (-fa 0) flag off.
//   Launch every arm through launch-stateos-tail-8099.ps1: it sets LONGSPEAR_PLE_HIST_REWIND=1 and
//   LONGSPEAR_PLE_HIST_LOG=1 in all of them, so the arms differ only in the tail lever. Check each
//   arm's log with `node tools/stateos-div-census.mjs --check step2 <log>` style gates: any
//   "[ple-hist] reset" at pos > 0 means a rewind the history repair missed.
//
// Usage:
//   node tools/stateos-tail-gate.mjs run --arm A0 --out <dir> [--url http://127.0.0.1:8099]
//        [--receipt <v2 receipt.json>] [--n-first 64] [--horizon 256] [--limit N]
//   node tools/stateos-tail-gate.mjs score --out <dir> [--lib <noise-floor-lib.mjs>]
//        [--shell C] [--benign A5,A3] [--min-prompts 20] [--n-min 6]
// The API key comes from LONGSPEAR_API_KEY, else from the --api-key line of
// D:/AI/llama-swap/config.yaml; it is never printed or written.

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath, pathToFileURL } from "node:url";

export const DEFAULT_RECEIPT =
  "D:/AI/worktrees/starfighter-trace-validation/.lanes/noise-floor-v2-20260908T022117Z/receipt.json";
export const DEFAULT_LIB =
  "D:/AI/worktrees/starfighter-trace-validation/.lanes/noise-floor/noise-floor-lib.mjs";
export const KEY_CONFIG = "D:/AI/llama-swap/config.yaml";

export function parseArgs(argv) {
  const o = {
    cmd: argv[0],
    arm: undefined,
    out: undefined,
    url: "http://127.0.0.1:8099",
    receipt: DEFAULT_RECEIPT,
    lib: DEFAULT_LIB,
    nFirst: 64,
    horizon: 256,
    limit: undefined,
    shell: "C",
    benign: ["A5", "A3"],
    minPrompts: 20,
    nMin: 6,
  };
  for (let i = 1; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = () => {
      i += 1;
      if (argv[i] === undefined) throw new Error(`${flag} needs a value`);
      return argv[i];
    };
    if (flag === "--arm") o.arm = value();
    else if (flag === "--out") o.out = path.resolve(value());
    else if (flag === "--url") o.url = value();
    else if (flag === "--receipt") o.receipt = value();
    else if (flag === "--lib") o.lib = value();
    else if (flag === "--n-first") o.nFirst = Number(value());
    else if (flag === "--horizon") o.horizon = Number(value());
    else if (flag === "--limit") o.limit = Number(value());
    else if (flag === "--shell") o.shell = value();
    else if (flag === "--benign") o.benign = value().split(",").filter(Boolean);
    else if (flag === "--min-prompts") o.minPrompts = Number(value());
    else if (flag === "--n-min") o.nMin = Number(value());
    else throw new Error(`unknown argument ${flag}`);
  }
  if (o.cmd !== "run" && o.cmd !== "score") throw new Error("first argument must be run or score");
  if (!o.out) throw new Error("--out is required");
  if (o.cmd === "run" && !/^[A-Za-z][A-Za-z0-9_-]*$/.test(o.arm ?? ""))
    throw new Error("--arm is required");
  if (!(o.nFirst >= 4) || !(o.horizon >= 1))
    throw new Error("--n-first must be >= 4 and --horizon >= 1");
  return o;
}

function apiKey() {
  if (process.env.LONGSPEAR_API_KEY) return process.env.LONGSPEAR_API_KEY;
  const m = /--api-key +"?([^" \r\n]+)/.exec(fs.readFileSync(KEY_CONFIG, "utf8"));
  if (!m) throw new Error(`no --api-key in ${KEY_CONFIG}`);
  return m[1];
}

export function loadPrompts(receiptPath) {
  const receipt = JSON.parse(fs.readFileSync(receiptPath, "utf8"));
  const prompts = receipt?.inputsEcho?.prompts;
  if (!Array.isArray(prompts) || prompts.length === 0)
    throw new Error(`no inputsEcho.prompts in ${receiptPath}`);
  return prompts.map((p) => {
    const text = fs.readFileSync(p.path, "utf8");
    const sha = crypto.createHash("sha256").update(fs.readFileSync(p.path)).digest("hex");
    if (sha !== p.sha256) throw new Error(`prompt ${p.id}: sha256 ${sha} != pinned ${p.sha256}`);
    return { id: p.id, sha256: sha, text };
  });
}

/** The forced prompt for request B: prompt + g[0..G-3] + [X], X a token that differs from g[G-2]. */
export function forcedPrompt(promptTokens, generated, candidates) {
  const G = generated.length;
  if (G < 2) return null;
  const original = generated[G - 2];
  const x = candidates.find((t) => t !== original);
  if (x === undefined) return null;
  return {
    tokens: [...promptTokens, ...generated.slice(0, G - 2), x],
    forcedIndex: promptTokens.length + G - 2,
    forcedToken: x,
    originalToken: original,
  };
}

async function post(url, key, route, body) {
  const res = await fetch(`${url}${route}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`${route} HTTP ${res.status}: ${text.slice(0, 300)}`);
  return text.length ? JSON.parse(text) : {};
}

function idsOf(completion) {
  const probs = completion.completion_probabilities;
  if (!Array.isArray(probs))
    throw new Error("no completion_probabilities (n_probs must be honoured)");
  return probs.map((p) => p.id);
}

const GREEDY = {
  temperature: 0,
  top_k: 1,
  top_p: 1,
  min_p: 0,
  n_probs: 1,
  cache_prompt: true,
  id_slot: 0,
  seed: 0,
};

async function runArm(o) {
  const key = apiKey();
  let prompts = loadPrompts(o.receipt);
  if (Number.isInteger(o.limit)) prompts = prompts.slice(0, o.limit);
  const candidates = (await post(o.url, key, "/tokenize", { content: "\n the" })).tokens ?? [];
  const record = {
    arm: o.arm,
    url: o.url,
    startedAt: new Date().toISOString(),
    receipt: o.receipt,
    nFirst: o.nFirst,
    horizon: o.horizon,
    prompts: [],
  };
  for (const p of prompts) {
    const row = { id: p.id, sha256: p.sha256, ok: false };
    try {
      try {
        await post(o.url, key, "/slots/0?action=erase", {});
        row.erased = true;
      } catch (e) {
        row.erased = false;
        row.eraseError = String(e.message ?? e).slice(0, 200);
      }
      const promptTokens = (await post(o.url, key, "/tokenize", { content: p.text })).tokens;
      const a = await post(o.url, key, "/completion", {
        ...GREEDY,
        prompt: promptTokens,
        n_predict: o.nFirst,
      });
      const generated = idsOf(a);
      const forced = forcedPrompt(promptTokens, generated, candidates);
      if (!forced) throw new Error(`request A generated ${generated.length} tokens; need >= 2`);
      const b = await post(o.url, key, "/completion", {
        ...GREEDY,
        prompt: forced.tokens,
        n_predict: o.horizon,
      });
      Object.assign(row, {
        ok: true,
        promptTokens: promptTokens.length,
        generated,
        forcedIndex: forced.forcedIndex,
        forcedToken: forced.forcedToken,
        originalToken: forced.originalToken,
        continuation: idsOf(b),
        stopA: a.stop_type ?? a.stopped_eos ?? null,
        stopB: b.stop_type ?? b.stopped_eos ?? null,
      });
    } catch (e) {
      row.error = String(e.message ?? e).slice(0, 300);
    }
    record.prompts.push(row);
    process.stdout.write(
      `${o.arm} ${p.id}: ${row.ok ? `ok (${row.continuation.length} tokens)` : `ERROR ${row.error}`}\n`,
    );
  }
  record.finishedAt = new Date().toISOString();
  fs.mkdirSync(o.out, { recursive: true });
  const file = path.join(o.out, `arm-${o.arm}.json`);
  fs.writeFileSync(file, `${JSON.stringify(record, null, 2)}\n`);
  process.stdout.write(`wrote ${file}\n`);
}

/** Pure scoring over arm records, mirroring the v2 campaign's per-prompt pairing. */
export function score(records, lib, { shell, benign, horizon, minPrompts, nMin }) {
  const byArm = new Map(records.map((r) => [r.arm, r]));
  for (const id of ["A0", "A1", shell, ...benign]) {
    if (!byArm.has(id)) throw new Error(`missing arm record arm-${id}.json`);
  }
  const tokensOf = (arm, promptId) => {
    const row = byArm.get(arm).prompts.find((p) => p.id === promptId);
    return row && row.ok ? row.continuation : null;
  };
  const promptIds = byArm.get("A0").prompts.map((p) => p.id);
  const rows = [];
  const dropped = [];
  let determinismFailures = 0;
  for (const id of promptIds) {
    const a0 = tokensOf("A0", id);
    const a1 = tokensOf("A1", id);
    if (!a0) {
      dropped.push({ id, reason: "A0 invalid" });
      continue;
    }
    const identical = Boolean(a1 && a1.length === a0.length && a1.every((v, i) => v === a0[i]));
    if (!identical) {
      determinismFailures += 1;
      dropped.push({ id, reason: a1 ? "determinism: A1 != A0" : "determinism: A1 unmeasured" });
      continue;
    }
    const eff = (arm) => {
      const t = tokensOf(arm, id);
      if (!t) return null;
      const m = lib.firstDivergence(a0, t);
      return m.valid ? lib.effectiveDivergence(m, horizon) : null;
    };
    const benignValues = benign.map(eff).filter((v) => v !== null);
    const s = eff(shell);
    if (benignValues.length < 2 || s === null) {
      dropped.push({ id, reason: s === null ? `${shell} invalid` : "envelope thin" });
      continue;
    }
    const worstBenign = Math.min(...benignValues);
    const delta = s - worstBenign;
    rows.push({
      id,
      shell: s,
      worstBenign,
      pairedDelta: delta,
      sign: delta < 0 ? "earlier" : delta > 0 ? "later" : "tie",
      envelopeSize: benignValues.length,
      sEarliest: benignValues.every((v) => s < v),
    });
  }
  const counts = { earlier: 0, tie: 0, later: 0 };
  for (const r of rows) counts[r.sign] += 1;
  const events = rows.filter((r) => Math.min(r.shell, r.worstBenign) < horizon);
  const earliest = events.filter((r) => r.sEarliest).length;
  const envelopeSize = rows.length ? Math.min(...rows.map((r) => r.envelopeSize)) : null;
  const delta = lib.medianIQR(rows.map((r) => r.pairedDelta));
  const rule = lib.classifyDivergence({
    validPrompts: rows.length,
    earlier: counts.earlier,
    tie: counts.tie,
    later: counts.later,
    events: events.length,
    medianDelta: delta.median,
    minPrompts,
    nMin,
    earliest,
    envelopeSize,
  });
  const voidGate = determinismFailures > 0;
  return {
    verdict: voidGate ? "void-determinism" : rule.outcome,
    voidReason: voidGate
      ? `A1 differed from A0 on ${determinismFailures} prompt(s): spec-on greedy is not deterministic; rerun with speculation off in all arms (merged plan section 4 step 4)`
      : null,
    rule,
    determinismFailures,
    dropped,
    rows,
  };
}

async function runScore(o) {
  const lib = await import(pathToFileURL(o.lib).href);
  const records = fs
    .readdirSync(o.out)
    .filter((f) => /^arm-.+\.json$/.test(f))
    .map((f) => JSON.parse(fs.readFileSync(path.join(o.out, f), "utf8")));
  const horizon = records[0]?.horizon ?? o.horizon;
  const result = score(records, lib, {
    shell: o.shell,
    benign: o.benign,
    horizon,
    minPrompts: o.minPrompts,
    nMin: o.nMin,
  });
  const file = path.join(o.out, "gate.json");
  fs.writeFileSync(
    file,
    `${JSON.stringify({ scoredAt: new Date().toISOString(), lib: o.lib, ...result }, null, 2)}\n`,
  );
  process.stdout.write(
    `verdict: ${result.verdict}${result.voidReason ? ` (${result.voidReason})` : ""}\n`,
  );
  process.stdout.write(`rule: ${result.rule.outcome} - ${result.rule.reason ?? ""}\n`);
  process.stdout.write(`wrote ${file}\n`);
}

const isEntryPoint =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isEntryPoint) {
  const o = parseArgs(process.argv.slice(2));
  (o.cmd === "run" ? runArm(o) : runScore(o)).catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
