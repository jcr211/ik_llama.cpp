#!/usr/bin/env node
// SV2-E1 W-SV2 step 4 driver: the standing v2 fidelity gate (docs/NOISE-FLOOR-V2-20260908.md,
// preregistered in docs/drafts/parity-noise-floor-20260906.md v2) applied to C1, the tail snapshot.
// COORDINATOR-RUN. It talks to an already-running llama-server; it never launches one.
//
// Per prompt (the 24 v2 prompts, read from the v2 campaign receipt and sha-checked):
//   1. erase slot 0 (best effort), tokenize the prompt. The erase is a NO-OP in W-SV2: /slots/:id is
//      registered only with --slot-save-path (server.cpp), which the launcher does not pass, so
//      row.erased is false and request A of prompt k+1 runs against prompt k's cache. Harmless here:
//      request A never chooses a tail (a tail sits at <= last cached - 2 of the PREVIOUS prompt's
//      generation, and A diverges near the prompt start), and A0, A1 and C take the same restore path
//      for A (same prompts in the same order); the tail writer's extra eviction cannot change C's list
//      below the 32-checkpoint cap. Every arm sends the same request sequence;
//   2. request A: greedy, max_tokens = --n-first, cache_prompt: true -> generated ids g[0..G-1].
//      After A the slot caches prompt + g[0..G-2] (the last sampled token is never decoded);
//   3. request B: the forced prompt, built ONCE from arm A0's request-A output:
//      prompt + g_A0[0..G-3] + [X], X != g_A0[G-2] by id and by text (also after deleting ' ', '\n',
//      '\r', which the server's text-level prefix match ignores). Every arm sends its OWN request A
//      (so its cache, and in arm C its tail, exist) and then A0's request B. On A0 and on any arm whose
//      request A matched A0's, B diverges at the last cached token (tail distance 1, the class C1
//      targets). Greedy continuation of --horizon tokens; the arm's record is B's continuation ids.
//   Token ids come from POST /v1/completions with logprobs: 1 (choices[0].logprobs.content[i].id,
//   examples/server/server-task.cpp to_json_oaicompat_final -> probs_vector_to_json); the id count
//   must equal usage.completion_tokens (a token that splits a UTF-8 character gets no logprobs entry).
//
// Arms (one server launch each through launch-stateos-tail-8099.ps1 with its own -LogStem; all with
// -DivLog; the launcher sets LONGSPEAR_PLE_HIST_REWIND=1 + LONGSPEAR_PLE_HIST_LOG=1 in every arm):
//   A0 flag off (RUN FIRST: the other arms read its record), A1 flag-off repeat (determinism control),
//   C = -Tail -DivLog, benign A5 (-ExtraArgs '-no-fmoe -no-fug') and A3 (-ExtraArgs '-fa 0') flag off.
//
// Scoring (needs every arm's record and server log, --arm-log ARM=path; each log's <LogStem>.flags
// sidecar, written by the launcher, records the arm's effective flags):
//   - a prompt is dropped when its B prompt (sha) differs across arms, when C's request-A output
//     differs from A0's, when C's response could not be parsed (e.g. a UTF-8-split id-count mismatch in
//     a drifted continuation; counted and reported), or when the request-B restore line of A0, A1 or C
//     is missing or lacks tail_dist=1, or C's is not a strict tail restore (chosen_origin=tail,
//     outcome=restored, reason=tail). Benign arms reach the same B from their own cache, so their
//     restore lines are not constrained. A restore line binds to a row by request order AND content:
//     n_past == the forced index, the cache window's marked token == g_A0[G-2] and the prompt window's
//     marked token == X;
//   - engagement: arm C must have >= --min-prompts strict tail restores on its request-B lines, else
//     `not-engaged`: VOID when tails were not available on enough B requests (no eligible prompts:
//     final rounds without an accepted draft), STOP when they were and C did not restore from them;
//   - before that: CUDA error lines in any arm's log = `cuda-errors` (STOP: stops the window);
//     an arm whose recorded flags differ from the required set (C: DIV_LOG + TAIL_SNAPSHOT; every other
//     arm: DIV_LOG only; all: PLE_HIST_REWIND + PLE_HIST_LOG; -ExtraArgs: A5 '-no-fmoe -no-fug', A3
//     '-fa 0', all others none), or with no flags record, = `mislaunched` (VOID) - decided from the
//     recorded flags only, never from the lever's own output; an HTTP or connection error on a C
//     request where A0's row succeeded = `shell-errors` (STOP);
//   - a C response that cannot be parsed is bound to its restore line with A0's row fields; after a
//     strict tail restore it counts as C diverging (unreadable output): more of them than the drop
//     slack (prompts - min-prompts) = `shell-unreadable` (STOP); up to the slack, a counted drop;
//   - A1's request A differing from A0's is spec-on nondeterminism: void-determinism, counted before
//     any drop;
//   - verdicts: compatible-at-horizon = PASS (exit 0); shellWorse, not-engaged:tails-not-used,
//     cuda-errors, shell-errors, shell-unreadable = STOP (exit 2); insufficient-sample, void-determinism, mislaunched and
//     not-engaged:no-eligible-prompts = VOID (exit 3): the gate did not answer, not a C1 kill.
//
// Usage:
//   node tools/stateos-tail-gate.mjs run --arm A0 --out <dir> [--url http://127.0.0.1:8099]
//        [--receipt <v2 receipt.json>] [--n-first 64] [--horizon 256] [--limit N]
//   node tools/stateos-tail-gate.mjs run --arm C --out <dir> [--reference <dir>/arm-A0.json] ...
//   node tools/stateos-tail-gate.mjs score --out <dir> --arm-log A0=<log> --arm-log A1=<log>
//        --arm-log C=<log> --arm-log A5=<log> --arm-log A3=<log> [--lib <noise-floor-lib.mjs>]
//        [--shell C] [--benign A5,A3] [--min-prompts 20] [--n-min 6]
// The API key comes from LONGSPEAR_API_KEY, else from the --api-key line of
// D:/AI/llama-swap/config.yaml; it is never printed or written.

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath, pathToFileURL } from "node:url";

import { REQUIRED_FLAGS, feedFiles, flagsMismatch } from "./stateos-div-census.mjs";

export const DEFAULT_RECEIPT =
  "D:/AI/worktrees/starfighter-trace-validation/.lanes/noise-floor-v2-20260908T022117Z/receipt.json";
export const DEFAULT_LIB =
  "D:/AI/worktrees/starfighter-trace-validation/.lanes/noise-floor/noise-floor-lib.mjs";
export const KEY_CONFIG = "D:/AI/llama-swap/config.yaml";
// single-token texts for the forced token X, tried in order
// (no whitespace-only text: the server's prefix match ignores ' ', '\n', '\r')
export const CANDIDATE_TEXTS = [" the", "0", "Z", ".", "X"];

export function parseArgs(argv) {
  const o = {
    cmd: argv[0],
    arm: undefined,
    out: undefined,
    url: "http://127.0.0.1:8099",
    receipt: DEFAULT_RECEIPT,
    reference: undefined,
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
    else if (flag === "--reference") o.reference = path.resolve(value());
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
  if (o.cmd === "run" && o.arm !== "A0" && o.reference === undefined)
    o.reference = path.join(o.out, "arm-A0.json");
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

export const tokensSha = (tokens) =>
  crypto.createHash("sha256").update(JSON.stringify(tokens)).digest("hex");

/**
 * Generated token ids and texts from a /v1/completions response (non-streamed, logprobs >= 1):
 * choices[0].logprobs.content[i] = {id, token, bytes, logprob, top_logprobs}. Throws unless every
 * id is an integer and the id count equals usage.completion_tokens. A completion with no generated
 * token has logprobs null and text "".
 */
export function tokensOf(resp) {
  const choice = resp?.choices?.[0];
  if (!choice) throw new Error("response has no choices[0]");
  const n = resp?.usage?.completion_tokens;
  if (!Number.isInteger(n)) throw new Error("response has no integer usage.completion_tokens");
  const content = choice.logprobs?.content;
  if (content == null) {
    if (choice.text === "" && n === 0) return { ids: [], texts: [] };
    throw new Error("choices[0].logprobs.content missing (logprobs must be requested)");
  }
  if (!Array.isArray(content)) throw new Error("choices[0].logprobs.content is not an array");
  const ids = content.map((e) => e?.id);
  if (!ids.every((id) => Number.isInteger(id)))
    throw new Error("a logprobs entry has no integer id");
  if (ids.length !== n) {
    throw new Error(
      `${ids.length} logprobs ids for ${n} completion tokens (a UTF-8-split token has no entry)`,
    );
  }
  return { ids, texts: content.map((e) => (typeof e?.token === "string" ? e.token : "")) };
}

// the server's non-exact prefix match ignores these characters (server-common.cpp text prefix)
const normalize = (s) => s.replace(/[ \n\r]/g, "");
const related = (a, b) => a.length > 0 && b.length > 0 && (a.startsWith(b) || b.startsWith(a));

/**
 * The forced prompt for request B: prompt + g[0..G-3] + [X]. X differs from g[G-2] by id and by text,
 * and neither text is a prefix of the other, raw or with ' ', '\n', '\r' deleted; X must also keep
 * some text after that deletion. So the server's text-level prefix match cannot absorb it.
 * candidates: [{id, text}] single-token candidates; originalText: the text of g[G-2].
 */
export function forcedPrompt(promptTokens, generated, candidates, originalText = "") {
  const G = generated.length;
  if (G < 2) return null;
  const original = generated[G - 2];
  const ok = (t) =>
    t.id !== original &&
    normalize(t.text).length > 0 &&
    !related(t.text, originalText) &&
    !related(normalize(t.text), normalize(originalText));
  const x = candidates.find(ok);
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

// error kinds recorded per failed row: "http" (HTTP status or connection error: the server failed) or
// "parse" (the response could not be turned into ids, e.g. an id-count mismatch)
async function complete(o, key, prompt, n) {
  let resp;
  try {
    resp = await post(o.url, key, "/v1/completions", {
      ...GREEDY,
      prompt,
      max_tokens: n,
      n_predict: n,
    });
  } catch (e) {
    e.kind = "http";
    throw e;
  }
  try {
    return tokensOf(resp);
  } catch (e) {
    e.kind = "parse";
    throw e;
  }
}

async function runArm(o) {
  const key = apiKey();
  let prompts = loadPrompts(o.receipt);
  if (Number.isInteger(o.limit)) prompts = prompts.slice(0, o.limit);
  const reference = o.arm === "A0" ? null : JSON.parse(fs.readFileSync(o.reference, "utf8"));
  if (reference && reference.arm !== "A0") throw new Error(`${o.reference} is not arm A0's record`);
  const candidates = [];
  if (!reference) {
    for (const text of CANDIDATE_TEXTS) {
      const t = (await post(o.url, key, "/tokenize", { content: text })).tokens ?? [];
      if (t.length === 1 && Number.isInteger(t[0])) candidates.push({ id: t[0], text });
    }
  }
  const record = {
    arm: o.arm,
    url: o.url,
    startedAt: new Date().toISOString(),
    receipt: o.receipt,
    reference: reference ? o.reference : null,
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
      let forced;
      if (reference) {
        const ref = reference.prompts.find((r) => r.id === p.id);
        if (!ref || !ref.ok) throw new Error("A0 has no valid row for this prompt");
        forced = {
          tokens: ref.bTokens,
          forcedIndex: ref.forcedIndex,
          forcedToken: ref.forcedToken,
          originalToken: ref.originalToken,
        };
      } else {
        forced = forcedPrompt(promptTokens, a.ids, candidates, a.texts[a.ids.length - 2] ?? "");
        if (!forced)
          throw new Error(
            `request A generated ${a.ids.length} tokens or no unrelated X; need >= 2`,
          );
      }
      const b = await complete(o, key, forced.tokens, o.horizon);
      Object.assign(row, {
        ok: true,
        promptTokens: promptTokens.length,
        generated: a.ids,
        forcedIndex: forced.forcedIndex,
        forcedToken: forced.forcedToken,
        originalToken: forced.originalToken,
        bSha: tokensSha(forced.tokens),
        ...(reference ? {} : { bTokens: forced.tokens }),
        continuation: b.ids,
      });
    } catch (e) {
      row.error = String(e.message ?? e).slice(0, 300);
      // "http": the server failed (status or connection); "parse": a response we could not read;
      // "input": no usable A0 row / no forced prompt. Plain errors come from post() -> "http".
      row.errorKind =
        e.kind ?? (/^A0 has no valid row|^request A generated/.test(row.error) ? "input" : "http");
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

// The live state after the restore is the tail's: outcome exactly "restored" with reason (the origin
// actually restored) "tail". Under the crosscheck the server continues on the flag-off checkpoint
// (outcome restored:xcheck-flag-off, reason = that checkpoint's origin): not a tail restore.
export const isTailRestore = (r) =>
  r.chosenOrigin === "tail" && r.outcome === "restored" && r.reason === "tail";

const centerOf = (w) => (w && w.center >= 0 ? w.ids[w.center] : undefined);

/** Does this restore line belong to the row's request B? Position AND content. */
export function bindsTo(row, r) {
  return (
    r.nPast === row.forcedIndex &&
    centerOf(r.cacheWin) === row.originalToken &&
    centerOf(r.promptWin) === row.forcedToken
  );
}

/**
 * Match each row of an arm record to its request-B restore line, in request order (the search
 * resumes after the previous match) and by content (bindsTo). Returns {promptId: restore | null}.
 * A row whose request B reached the server but whose response could not be parsed (errorKind "parse")
 * is bound with the reference (A0) row's fields: it sent A0's B prompt, so the forced position and the
 * two marked tokens are A0's.
 */
export function matchRestores(record, restores, reference = null) {
  const out = {};
  let j = 0;
  for (const row of record.prompts) {
    let key = row.ok ? row : null;
    if (!row.ok && row.errorKind === "parse" && reference) {
      const ref = reference.prompts.find((p) => p.id === row.id);
      if (ref?.ok) key = ref;
    }
    if (!key) {
      out[row.id] = null;
      continue;
    }
    let k = j;
    while (k < restores.length && !bindsTo(key, restores[k])) k += 1;
    if (k < restores.length) {
      out[row.id] = restores[k];
      j = k + 1;
    } else {
      out[row.id] = null;
    }
  }
  return out;
}

const sameIds = (a, b) =>
  Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => v === b[i]);

/**
 * Pre-scoring filter: engagement of the shell arm and per-prompt drops.
 * restoresByArm: {arm: census restore rows (in log order)}.
 */
export function engagementAndDrops(records, restoresByArm, { shell, minPrompts }) {
  const byArm = new Map(records.map((r) => [r.arm, r]));
  const a0 = byArm.get("A0");
  if (!a0) throw new Error("missing arm record arm-A0.json");
  const constrained = ["A0", "A1", shell].filter((a) => byArm.has(a));
  const matched = {};
  for (const a of constrained) matched[a] = matchRestores(byArm.get(a), restoresByArm[a] ?? [], a0);
  const drop = new Map();
  const droppedCounts = {};
  const note = (id, reason) => {
    if (!drop.has(id)) drop.set(id, reason);
    droppedCounts[reason] = (droppedCounts[reason] ?? 0) + 1;
  };
  const rowOf = (arm, id) => byArm.get(arm)?.prompts.find((p) => p.id === id);
  let tailRestores = 0;
  let tailAvailableAtB = 0;
  let unreadableAfterTail = 0;
  for (const p of a0.prompts) {
    const id = p.id;
    // one B prompt for every arm
    for (const r of records) {
      const row = rowOf(r.arm, id);
      if (row?.ok && p.ok && row.bSha !== p.bSha) note(id, `${r.arm}: request B differs from A0's`);
    }
    // C's cache must equal A0's. A1's request A differing from A0's is NOT a drop: it is spec-on greedy
    // nondeterminism, which `score` turns into void-determinism (so A1's restore line is not
    // constrained for that prompt either).
    const cRow = rowOf(shell, id);
    if (cRow?.ok && p.ok && !sameIds(cRow.generated, p.generated)) {
      note(id, `${shell}: request A output differs from A0's`);
    }
    // a C response we could not read (e.g. a UTF-8-split id-count mismatch) is not a server failure
    // (HTTP/connection errors are STOP in preScoreChecks). After a strict tail restore on a B that A0
    // read fine it is evidence that C DIVERGED (corrupted output): counted as unreadable-after-tail;
    // more than the drop slack (prompts - minPrompts) of them is STOP (preVerdict). Up to the slack it
    // is a counted drop (benign drift).
    const cParseFailed = Boolean(cRow && !cRow.ok && cRow.errorKind === "parse" && p.ok);
    if (cParseFailed) {
      note(id, `${shell}: response not parsed`);
      const m = matched[shell]?.[id];
      if (m && isTailRestore(m)) unreadableAfterTail += 1;
    }
    const a1Row = rowOf("A1", id);
    const a1Diverged = Boolean(a1Row?.ok && p.ok && !sameIds(a1Row.generated, p.generated));
    for (const a of constrained) {
      if (a === "A1" && a1Diverged) continue;
      const m = matched[a][id];
      if (!m) note(id, `${a}: no request-B restore line`);
      else if (m.tailDist !== 1) note(id, `${a}: tail_dist=${m.tailDist}`);
      else if (a === shell && !isTailRestore(m))
        note(id, `${shell}: not restored from the tail (${m.chosenOrigin}/${m.outcome})`);
    }
    const mc = matched[shell]?.[id];
    if (mc) {
      tailAvailableAtB += mc.tailAvailable ? 1 : 0;
      tailRestores += isTailRestore(mc) ? 1 : 0;
    }
  }
  const engaged = tailRestores >= minPrompts;
  const slack = Math.max(0, a0.prompts.length - minPrompts);
  return {
    engaged,
    tailRestores,
    tailAvailableAtB,
    eligibleExisted: tailAvailableAtB >= minPrompts,
    unreadableAfterTail,
    slack,
    drop,
    droppedCounts,
    droppedPrompts: drop.size,
  };
}

/**
 * The verdict before scoring, from engagementAndDrops' result, or null to score.
 *   - C responses unreadable after a strict tail restore beyond the drop slack: `shell-unreadable` (STOP)
 *   - not engaged: `not-engaged:tails-not-used` (STOP) or `not-engaged:no-eligible-prompts` (VOID)
 */
export function preVerdict(pre, { shell, minPrompts }) {
  if (pre.unreadableAfterTail > pre.slack) {
    return {
      verdict: "shell-unreadable",
      voidReason: `${pre.unreadableAfterTail} ${shell} responses could not be read after a strict tail restore on a B that A0 read (drop slack ${pre.slack}): the tail-restored state produced unreadable output`,
    };
  }
  if (pre.engaged) return null;
  if (pre.eligibleExisted) {
    return {
      verdict: "not-engaged:tails-not-used",
      voidReason: `tails were available on ${pre.tailAvailableAtB} request-B divergences in ${shell} but restored on only ${pre.tailRestores} (< ${minPrompts}): the lever failed where it could act`,
    };
  }
  return {
    verdict: "not-engaged:no-eligible-prompts",
    voidReason: `tails were available on only ${pre.tailAvailableAtB} request-B divergences in ${shell} (< ${minPrompts}; final rounds without an accepted draft leave no tail): no eligible prompts, the gate did not answer`,
  };
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
  const rowOf = (arm, promptId) => byArm.get(arm).prompts.find((p) => p.id === promptId);
  const contOf = (arm, promptId) => {
    const row = rowOf(arm, promptId);
    return row && row.ok ? row.continuation : null;
  };
  const promptIds = byArm.get("A0").prompts.map((p) => p.id);
  const rows = [];
  const dropped = [];
  let determinismFailures = 0;
  for (const id of promptIds) {
    // determinism covers both requests, and is counted BEFORE any drop so it is never hidden (e.g. when
    // A0 is the odd one out and C's request A differs too): A1's request A must reproduce A0's
    const a0RowA = rowOf("A0", id);
    const a1RowA = rowOf("A1", id);
    if (
      a0RowA?.ok &&
      a1RowA?.ok &&
      Array.isArray(a0RowA.generated) &&
      !sameIds(a1RowA.generated, a0RowA.generated)
    ) {
      determinismFailures += 1;
      dropped.push({ id, reason: "determinism: A1 request A != A0" });
      continue;
    }
    if (drop.has(id)) {
      dropped.push({ id, reason: drop.get(id) });
      continue;
    }
    // every arm must have answered the same B prompt
    const shas = new Set(
      ["A0", "A1", shell, ...benign]
        .map((a) => rowOf(a, id))
        .filter((r) => r?.ok)
        .map((r) => r.bSha ?? null),
    );
    if (shas.size > 1 || shas.has(null)) {
      dropped.push({ id, reason: "request B differs across arms" });
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

/** PASS | STOP | VOID for a gate verdict (VOID: the gate did not answer; not a C1 kill). */
export function gateStatus(verdict) {
  if (verdict === "compatible-at-horizon") return "PASS";
  if (
    verdict === "shellWorse" ||
    verdict === "not-engaged:tails-not-used" ||
    verdict === "cuda-errors" ||
    verdict === "shell-errors" ||
    verdict === "shell-unreadable"
  ) {
    return "STOP";
  }
  return "VOID";
}

/** Required recorded flags per step-4 arm (env flags and -ExtraArgs). */
export function requiredFlagsOfArm(arm, shell) {
  if (arm === shell) return REQUIRED_FLAGS.gateTail;
  if (arm === "A5") return REQUIRED_FLAGS.gateA5;
  if (arm === "A3") return REQUIRED_FLAGS.gateA3;
  return REQUIRED_FLAGS.gateOff;
}

/**
 * Checks on the arms' logs and records before any scoring. censusByArm: {arm: raw census state
 * (feedFiles: the <LogStem>.flags sidecar + the log)}. Returns a verdict object, or null to proceed to
 * engagement and scoring.
 *   - any CUDA error line in any arm's log: `cuda-errors` (STOP: stops the window);
 *   - an arm whose RECORDED flags differ from its required set (C: REQUIRED_FLAGS.gateTail, others:
 *     REQUIRED_FLAGS.gateOff), or with no flags record: `mislaunched` (VOID). Decided only from the
 *     launcher's record, never from the lever's own output;
 *   - an HTTP or connection error on a C request where A0's row is ok (the server failed on C only):
 *     `shell-errors` (STOP). A C response that could not be parsed is a drop, not a STOP.
 */
export function preScoreChecks(records, censusByArm, { shell }) {
  const errors = Object.entries(censusByArm)
    .filter(([, c]) => c.cudaErrors > 0)
    .map(([arm, c]) => `${arm}=${c.cudaErrors}`);
  if (errors.length) {
    return {
      verdict: "cuda-errors",
      voidReason: `CUDA error lines in arm logs: ${errors.join(", ")}`,
    };
  }
  const mislaunched = records
    .map((r) => {
      const c = censusByArm[r.arm];
      const why = flagsMismatch(
        c?.flags ?? null,
        requiredFlagsOfArm(r.arm, shell),
        c?.flagsProblem ?? null,
      );
      return why ? `${r.arm}: ${why}` : null;
    })
    .filter(Boolean);
  if (mislaunched.length) {
    return {
      verdict: "mislaunched",
      voidReason: `recorded launch flags do not match the step-4 arms: ${mislaunched.join("; ")}`,
    };
  }
  const a0 = records.find((r) => r.arm === "A0");
  const cRec = records.find((r) => r.arm === shell);
  if (a0 && cRec) {
    const failed = a0.prompts
      .filter((p) => {
        const q = cRec.prompts.find((x) => x.id === p.id);
        return p.ok && q && !q.ok && (q.errorKind ?? "http") === "http";
      })
      .map((p) => p.id);
    if (failed.length) {
      return {
        verdict: "shell-errors",
        voidReason: `arm ${shell} failed on ${failed.length} prompt(s) where A0 succeeded: ${failed.join(", ")}`,
      };
    }
  }
  return null;
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
  const censusByArm = {};
  const restoresByArm = {};
  for (const r of records) {
    // the log plus its <LogStem>.flags sidecar (the arm's recorded launch flags)
    censusByArm[r.arm] = await feedFiles([o.armLogs[r.arm]]);
    restoresByArm[r.arm] = censusByArm[r.arm].v2.restores;
  }
  const pre = engagementAndDrops(records, restoresByArm, {
    shell: o.shell,
    minPrompts: o.minPrompts,
  });
  const horizon = records[0]?.horizon ?? o.horizon;
  const refused =
    preScoreChecks(records, censusByArm, { shell: o.shell }) ??
    preVerdict(pre, { shell: o.shell, minPrompts: o.minPrompts });
  const result = refused
    ? { ...refused, rule: { outcome: "not-scored", reason: "" } }
    : score(records, lib, {
        shell: o.shell,
        benign: o.benign,
        horizon,
        minPrompts: o.minPrompts,
        nMin: o.nMin,
        drop: pre.drop,
      });
  const status = gateStatus(result.verdict);
  const file = path.join(o.out, "gate.json");
  const report = {
    scoredAt: new Date().toISOString(),
    lib: o.lib,
    armLogs: o.armLogs,
    status,
    engagement: {
      tailRestores: pre.tailRestores,
      tailAvailableAtB: pre.tailAvailableAtB,
      required: o.minPrompts,
      engaged: pre.engaged,
      unreadableAfterTail: pre.unreadableAfterTail,
      slack: pre.slack,
    },
    droppedPrompts: pre.droppedPrompts,
    droppedCounts: pre.droppedCounts,
    ...result,
  };
  fs.writeFileSync(file, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(
    `engagement: ${pre.tailRestores} tail restores, tails available on ${pre.tailAvailableAtB} request-B divergences in ${o.shell} (need ${o.minPrompts})\n`,
  );
  process.stdout.write(
    `dropped prompts: ${pre.droppedPrompts} ${JSON.stringify(pre.droppedCounts)}\n`,
  );
  process.stdout.write(
    `verdict: ${result.verdict}${result.voidReason ? ` (${result.voidReason})` : ""}\n`,
  );
  process.stdout.write(`rule: ${result.rule.outcome} - ${result.rule.reason ?? ""}\n`);
  process.stdout.write(
    `status: ${status}${status === "VOID" ? " (the gate did not answer; not a C1 kill)" : ""}\n`,
  );
  process.stdout.write(`wrote ${file}\n`);
  process.exitCode = status === "PASS" ? 0 : status === "VOID" ? 3 : 2;
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
