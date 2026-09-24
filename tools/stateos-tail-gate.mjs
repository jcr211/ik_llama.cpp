#!/usr/bin/env node
// SV2-E1 W-SV2 step 4 driver: the standing v2 fidelity gate (docs/NOISE-FLOOR-V2-20260908.md,
// preregistered in docs/drafts/parity-noise-floor-20260906.md v2) applied to C1, the tail snapshot.
// COORDINATOR-RUN. It talks to an already-running llama-server; it never launches one.
//
// Per prompt (the 24 v2 prompts, read from the v2 campaign receipt and sha-checked):
//   1. erase slot 0 (best effort: needs --slot-save-path), tokenize the prompt;
//   2. request A: greedy, max_tokens = --n-first, cache_prompt: true -> generated ids g[0..G-1].
//      After A the slot normally caches prompt + g[0..G-2] (the last sampled token is never decoded);
//   3. request B: prompt + g[0..G-3] + [X], X != g[G-2] and textually unrelated to it: the prompt
//      diverges at the last cached token (tail distance 1, the class C1 targets); greedy
//      continuation of --horizon tokens.
//   Token ids come from POST /v1/completions with logprobs: 1 (choices[0].logprobs.content[i].id,
//   examples/server/server-task.cpp to_json_oaicompat_final -> probs_vector_to_json). The fork's
//   /completion carries no ids (its completion_probabilities entries are {content, probs}).
//   The arm's record is B's continuation ids.
//
// Arms (one server launch each, speculation config identical, flags per arm; ALL with -DivLog so
// every arm's log shows where the forced divergence landed):
//   A0 flag off, A1 flag-off repeat (fresh process, determinism control), C = -Tail -DivLog,
//   benign A5 (-ExtraArgs '-no-fmoe -no-fug') and A3 (-ExtraArgs '-fa 0') flag off.
//   Launch every arm through launch-stateos-tail-8099.ps1 with its own -LogStem: it sets
//   LONGSPEAR_PLE_HIST_REWIND=1 and LONGSPEAR_PLE_HIST_LOG=1 in all of them, so the arms differ only
//   in the tail lever.
//
// Scoring needs every arm's server log (--arm-log ARM=path). Before scoring:
//   - engagement: arm C's census must show >= --min-prompts restores that chose the tail and restored
//     (chosen_origin=tail, outcome restored*); otherwise the verdict is `not-engaged`;
//   - per prompt, request B's [stateos-div] restore line is found in each arm's log (n_past == the
//     forced index). A prompt is dropped when any arm's line is missing or lacks tail_dist=1, or when
//     arm C's line lacks chosen_origin=tail with outcome restored*. Dropped counts are reported.
//
// Usage:
//   node tools/stateos-tail-gate.mjs run --arm A0 --out <dir> [--url http://127.0.0.1:8099]
//        [--receipt <v2 receipt.json>] [--n-first 64] [--horizon 256] [--limit N]
//   node tools/stateos-tail-gate.mjs score --out <dir> --arm-log A0=<log> --arm-log A1=<log>
//        --arm-log C=<log> --arm-log A5=<log> --arm-log A3=<log> [--lib <noise-floor-lib.mjs>]
//        [--shell C] [--benign A5,A3] [--min-prompts 20] [--n-min 6]
// The API key comes from LONGSPEAR_API_KEY, else from the --api-key line of
// D:/AI/llama-swap/config.yaml; it is never printed or written.

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import readline from "node:readline";
import { fileURLToPath, pathToFileURL } from "node:url";

import { feed, newCensus } from "./stateos-div-census.mjs";

export const DEFAULT_RECEIPT =
  "D:/AI/worktrees/starfighter-trace-validation/.lanes/noise-floor-v2-20260908T022117Z/receipt.json";
export const DEFAULT_LIB =
  "D:/AI/worktrees/starfighter-trace-validation/.lanes/noise-floor/noise-floor-lib.mjs";
export const KEY_CONFIG = "D:/AI/llama-swap/config.yaml";
// single-token texts for the forced token X, tried in order
export const CANDIDATE_TEXTS = ["\n", " the", "0", "Z", "."];

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
    armLogs: {},
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
    else if (flag === "--arm-log") {
      const v = value();
      const eq = v.indexOf("=");
      if (eq <= 0) throw new Error("--arm-log needs ARM=path");
      o.armLogs[v.slice(0, eq)] = v.slice(eq + 1);
    } else throw new Error(`unknown argument ${flag}`);
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

/**
 * Generated token ids and texts from a /v1/completions response (non-streamed, logprobs >= 1):
 * choices[0].logprobs.content[i] = {id, token, bytes, logprob, top_logprobs}. Throws unless every
 * id is an integer. A completion with no generated token has logprobs null and text "".
 */
export function tokensOf(resp) {
  const choice = resp?.choices?.[0];
  if (!choice) throw new Error("response has no choices[0]");
  const content = choice.logprobs?.content;
  if (content == null) {
    if (choice.text === "") return { ids: [], texts: [] };
    throw new Error("choices[0].logprobs.content missing (logprobs must be requested)");
  }
  if (!Array.isArray(content)) throw new Error("choices[0].logprobs.content is not an array");
  const ids = content.map((e) => e?.id);
  if (!ids.every((id) => Number.isInteger(id)))
    throw new Error("a logprobs entry has no integer id");
  return { ids, texts: content.map((e) => (typeof e?.token === "string" ? e.token : "")) };
}

/**
 * The forced prompt for request B: prompt + g[0..G-3] + [X]. X differs from g[G-2] by id and by text
 * (neither text a prefix of the other), so the server's text-level prefix match cannot absorb it.
 * candidates: [{id, text}] single-token candidates; originalText: the text of g[G-2].
 */
export function forcedPrompt(promptTokens, generated, candidates, originalText = "") {
  const G = generated.length;
  if (G < 2) return null;
  const original = generated[G - 2];
  const unrelated = (t) =>
    t.id !== original &&
    t.text.length > 0 &&
    !(
      originalText.length > 0 &&
      (originalText.startsWith(t.text) || t.text.startsWith(originalText))
    );
  const x = candidates.find(unrelated);
  if (x === undefined) return null;
  return {
    tokens: [...promptTokens, ...generated.slice(0, G - 2), x.id],
    forcedIndex: promptTokens.length + G - 2,
    forcedToken: x.id,
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

const GREEDY = {
  temperature: 0,
  top_k: 1,
  top_p: 1,
  min_p: 0,
  logprobs: 1,
  cache_prompt: true,
  id_slot: 0,
  seed: 0,
  stream: false,
};

async function complete(o, key, prompt, n) {
  return tokensOf(
    await post(o.url, key, "/v1/completions", { ...GREEDY, prompt, max_tokens: n, n_predict: n }),
  );
}

async function runArm(o) {
  const key = apiKey();
  let prompts = loadPrompts(o.receipt);
  if (Number.isInteger(o.limit)) prompts = prompts.slice(0, o.limit);
  const candidates = [];
  for (const text of CANDIDATE_TEXTS) {
    const t = (await post(o.url, key, "/tokenize", { content: text })).tokens ?? [];
    if (t.length === 1 && Number.isInteger(t[0])) candidates.push({ id: t[0], text });
  }
  const record = {
    arm: o.arm,
    url: o.url,
    startedAt: new Date().toISOString(),
    receipt: o.receipt,
    nFirst: o.nFirst,
    horizon: o.horizon,
    candidates,
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
      const a = await complete(o, key, promptTokens, o.nFirst);
      const forced = forcedPrompt(promptTokens, a.ids, candidates, a.texts[a.ids.length - 2] ?? "");
      if (!forced)
        throw new Error(`request A generated ${a.ids.length} tokens or no unrelated X; need >= 2`);
      const b = await complete(o, key, forced.tokens, o.horizon);
      Object.assign(row, {
        ok: true,
        promptTokens: promptTokens.length,
        generated: a.ids,
        forcedIndex: forced.forcedIndex,
        forcedToken: forced.forcedToken,
        originalToken: forced.originalToken,
        continuation: b.ids,
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

const isTailRestore = (r) => r.chosenOrigin === "tail" && String(r.outcome).startsWith("restored");

/**
 * Match each ok row of an arm record to its request-B restore line (in log order, n_past == the
 * forced index). Returns {promptId: restore | null}.
 */
export function matchRestores(record, restores) {
  const out = {};
  let j = 0;
  for (const row of record.prompts) {
    if (!row.ok) {
      out[row.id] = null;
      continue;
    }
    let k = j;
    while (k < restores.length && restores[k].nPast !== row.forcedIndex) k += 1;
    if (k < restores.length) {
      out[row.id] = restores[k];
      j = k + 1;
    } else {
      out[row.id] = null;
    }
  }
  return out;
}

/**
 * Pre-scoring filter: engagement of the shell arm and per-prompt drops.
 * restoresByArm: {arm: census restore rows (in log order)}.
 */
export function engagementAndDrops(records, restoresByArm, { shell, minPrompts }) {
  const shellRestores = restoresByArm[shell] ?? [];
  const tailRestores = shellRestores.filter(isTailRestore).length;
  const engaged = tailRestores >= minPrompts;
  const matched = {};
  for (const r of records) matched[r.arm] = matchRestores(r, restoresByArm[r.arm] ?? []);
  const promptIds = (records.find((r) => r.arm === "A0") ?? records[0]).prompts.map((p) => p.id);
  const drop = new Map();
  const droppedCounts = {};
  const note = (id, reason) => {
    if (!drop.has(id)) drop.set(id, reason);
    droppedCounts[reason] = (droppedCounts[reason] ?? 0) + 1;
  };
  for (const id of promptIds) {
    for (const r of records) {
      const m = matched[r.arm][id];
      if (!m) note(id, `${r.arm}: no restore line`);
      else if (m.tailDist !== 1) note(id, `${r.arm}: tail_dist=${m.tailDist}`);
      else if (r.arm === shell && !isTailRestore(m))
        note(id, `${shell}: not restored from the tail (${m.chosenOrigin}/${m.outcome})`);
    }
  }
  return { engaged, tailRestores, drop, droppedCounts, droppedPrompts: drop.size };
}

/** Pure scoring over arm records, mirroring the v2 campaign's per-prompt pairing. */
export function score(
  records,
  lib,
  { shell, benign, horizon, minPrompts, nMin, drop = new Map() },
) {
  const byArm = new Map(records.map((r) => [r.arm, r]));
  for (const id of ["A0", "A1", shell, ...benign]) {
    if (!byArm.has(id)) throw new Error(`missing arm record arm-${id}.json`);
  }
  const contOf = (arm, promptId) => {
    const row = byArm.get(arm).prompts.find((p) => p.id === promptId);
    return row && row.ok ? row.continuation : null;
  };
  const promptIds = byArm.get("A0").prompts.map((p) => p.id);
  const rows = [];
  const dropped = [];
  let determinismFailures = 0;
  for (const id of promptIds) {
    if (drop.has(id)) {
      dropped.push({ id, reason: drop.get(id) });
      continue;
    }
    const a0 = contOf("A0", id);
    const a1 = contOf("A1", id);
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
      const t = contOf(arm, id);
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

async function restoresOfLog(file) {
  const c = newCensus();
  const rl = readline.createInterface({ input: fs.createReadStream(file), crlfDelay: Infinity });
  for await (const line of rl) feed(c, line);
  return c.v2.restores;
}

async function runScore(o) {
  const lib = await import(pathToFileURL(o.lib).href);
  const records = fs
    .readdirSync(o.out)
    .filter((f) => /^arm-.+\.json$/.test(f))
    .map((f) => JSON.parse(fs.readFileSync(path.join(o.out, f), "utf8")));
  for (const r of records) {
    if (!o.armLogs[r.arm])
      throw new Error(`--arm-log ${r.arm}=<server log> is required for every arm`);
  }
  const restoresByArm = {};
  for (const r of records) restoresByArm[r.arm] = await restoresOfLog(o.armLogs[r.arm]);
  const pre = engagementAndDrops(records, restoresByArm, {
    shell: o.shell,
    minPrompts: o.minPrompts,
  });
  const horizon = records[0]?.horizon ?? o.horizon;
  const result = pre.engaged
    ? score(records, lib, {
        shell: o.shell,
        benign: o.benign,
        horizon,
        minPrompts: o.minPrompts,
        nMin: o.nMin,
        drop: pre.drop,
      })
    : {
        verdict: "not-engaged",
        voidReason: `arm ${o.shell} restored from a tail on ${pre.tailRestores} requests (< ${o.minPrompts}): the lever did not engage; not scored`,
        rule: { outcome: "not-scored", reason: "" },
      };
  const file = path.join(o.out, "gate.json");
  const report = {
    scoredAt: new Date().toISOString(),
    lib: o.lib,
    armLogs: o.armLogs,
    engagement: { tailRestores: pre.tailRestores, required: o.minPrompts, engaged: pre.engaged },
    droppedPrompts: pre.droppedPrompts,
    droppedCounts: pre.droppedCounts,
    ...result,
  };
  fs.writeFileSync(file, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(
    `engagement: ${pre.tailRestores} tail restores in ${o.shell} (need ${o.minPrompts})\n`,
  );
  process.stdout.write(
    `dropped prompts: ${pre.droppedPrompts} ${JSON.stringify(pre.droppedCounts)}\n`,
  );
  process.stdout.write(
    `verdict: ${result.verdict}${result.voidReason ? ` (${result.voidReason})` : ""}\n`,
  );
  process.stdout.write(`rule: ${result.rule.outcome} - ${result.rule.reason ?? ""}\n`);
  process.stdout.write(`wrote ${file}\n`);
  if (result.verdict !== "compatible-at-horizon") process.exitCode = 2;
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
