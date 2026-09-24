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
//      below the 32-checkpoint cap. Every arm sends the same request sequence. Caveat: the ngram-mod
//      draft table is server-lifetime shared state, so after a harmless B divergence C's later
//      request-A drafts can differ from A0's; outputs stay greedy-exact in principle (drafts affect
//      speed only), and if they do not, the determinism control (A1 vs A0) returns void-determinism;
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
// sidecar, written by the launcher, records the arm's effective flags, and its <LogStem>.port sidecar,
// written by check-stateos-port-8099.ps1, the listener check):
//   - log checks first (preScoreChecks): CUDA error lines in any arm's log = `cuda-errors` (STOP);
//     an arm whose recorded flags differ from the required set (C: DIV_LOG + TAIL_SNAPSHOT; every other
//     arm: DIV_LOG only; all: PLE_HIST_REWIND + PLE_HIST_LOG; -ExtraArgs: A5 '-no-fmoe -no-fug', A3
//     '-fa 0', all others none), or with no flags record, = `mislaunched` (VOID) - decided from the
//     recorded flags only, never from the lever's own output; a `[ple-hist] reset` at pos > 0 in any
//     arm = `ple-hist` (STOP); a <LogStem>.port record that is missing or does not name the model port
//     with exactly one listener PID equal to <LogStem>.pid = `mislaunched` (VOID; log lines are never
//     port evidence); no `[ple-hist] set` line in an arm = `ple-hist` (STOP);
//   - missing arm records (A0, A1, C, A5, A3 are all required): refuse with an error (exit 1);
//   - then the determinism control, BEFORE any C attribution: A1 differing from A0 on ANY prompt where
//     both rows are valid (request A ids, or B's continuation) = `void-determinism` (VOID);
//   - one validity rule (rowProblem) for engagement, determinism and scoring: a failed request, or a
//     successful one with fewer than 2 request-A / 1 request-B tokens, is an invalid row;
//   - every prompt is scorable or excluded once (engagementAndDrops). Invalid A0/A1 rows are control
//     failures, checked before C and never charged to C; more of them than the slack, with too few
//     scorable prompts left, = `insufficient-control` (VOID). A restore line binds to a row by
//     request order AND content: n_past == the forced index, the cache window's marked token ==
//     g_A0[G-2] and the prompt window's marked token == X. A C row whose request B reached the server
//     but failed is bound with A0's row fields. Benign arms reach the same B from their own cache, so
//     their restore lines are not constrained;
//   - ONE shared slack (prompts - min-prompts, 24 - 20 = 4) for every exclusion attributable to arm C,
//     whatever the reason: C's request A differs from A0's (where A1 reproduced A0), C's row is invalid
//     (request A or B not parsed, HTTP/connection error, too few tokens), C's B differs, or, on a prompt A0's
//     line shows as eligible (prev_round=drafted, prev_n_acc >= 1), C's request-B restore line is
//     missing, not at tail_dist=1, or not a strict tail restore (chosen_origin=tail, outcome=restored,
//     reason=tail). Each is reported with its prompt, request (A or B) and reason. More than the slack =
//     `shell-diverged` (STOP); up to the slack they are counted drops (tolerated by design);
//   - exclusions NOT attributable to C (A0 or A1 problems, A0 showing no eligible tail, a benign arm's
//     different B) leaving fewer than --min-prompts scorable strict tail restores =
//     `not-engaged:no-eligible-prompts` (VOID);
//   - verdicts: compatible-at-horizon = PASS (exit 0); shellWorse, cuda-errors, ple-hist,
//     shell-diverged = STOP (exit 2); insufficient-sample, void-determinism, mislaunched,
//     insufficient-control and not-engaged:no-eligible-prompts = VOID (exit 3): the gate did not
//     answer, not a C1 kill. A missing arm record or any other refusal to run = exit 1.
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

import { REQUIRED_FLAGS, feedFiles, flagsMismatch, portProblemOf } from "./stateos-div-census.mjs";

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
    // which request failed, if any: "A" (tokenize or request A) or "B" (request B was sent)
    let stage = "A";
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
      row.generated = a.ids; // kept even when request B fails
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
      row.bSha = tokensSha(forced.tokens);
      stage = "B";
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
      row.failedAt = stage;
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
 * A row whose request B reached the server but failed (failedAt "B"; old records: errorKind "parse")
 * is bound with the reference (A0) row's fields: it sent A0's B prompt, so the forced position and the
 * two marked tokens are A0's.
 */
export function matchRestores(record, restores, reference = null) {
  const out = {};
  let j = 0;
  for (const row of record.prompts) {
    let key = row.ok ? row : null;
    // a failed row binds (through the reference row) only if its request B reached the server
    const sentB = row.failedAt ? row.failedAt === "B" : row.errorKind === "parse";
    if (!row.ok && sentB && reference) {
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

// request A must return >= 2 tokens (B replaces the second-to-last one); request B >= 1 (the v2 lib's
// firstDivergence is invalid on an empty continuation)
export const MIN_A_TOKENS = 2;
export const MIN_B_TOKENS = 1;

/**
 * THE validity rule for an arm's row, shared by engagementAndDrops, the determinism check and score
 * (so score never rejects a continuation that engagement counted as valid). null when valid, else
 * {request: "A" | "B", reason, detail}. A successful response with too few tokens is invalid.
 */
export function rowProblem(row) {
  if (!row) return { request: "A", reason: "no row in the arm record", detail: "" };
  if (!row.ok) {
    const request = row.failedAt ?? (row.errorKind === "parse" ? "B" : "A");
    const what =
      row.errorKind === "parse"
        ? "not parsed"
        : row.errorKind === "input"
          ? "not sent (input error)"
          : "HTTP/connection error";
    return {
      request,
      reason: `request ${request} ${what}`,
      detail: String(row.error ?? "").slice(0, 120),
    };
  }
  const nA = Array.isArray(row.generated) ? row.generated.length : 0;
  if (nA < MIN_A_TOKENS) {
    return {
      request: "A",
      reason: `request A returned < ${MIN_A_TOKENS} tokens`,
      detail: `${nA} tokens`,
    };
  }
  const nB = Array.isArray(row.continuation) ? row.continuation.length : 0;
  if (nB < MIN_B_TOKENS) {
    return {
      request: "B",
      reason: `request B returned < ${MIN_B_TOKENS} tokens`,
      detail: `${nB} tokens`,
    };
  }
  return null;
}

/**
 * Determinism control (merged plan section 4 step 4): A1 must reproduce A0 on every prompt where both
 * rows are valid - request A's ids, and B's continuation where both sent the same B. Returns the
 * failing prompts [{id, request}]. Checked BEFORE any C attribution or slack: spec-on nondeterminism
 * would otherwise show up as C differences.
 */
export function determinismFailures(records) {
  const a0 = records.find((r) => r.arm === "A0");
  const a1 = records.find((r) => r.arm === "A1");
  if (!a0 || !a1) return [];
  const out = [];
  for (const p of a0.prompts) {
    const q = a1.prompts.find((x) => x.id === p.id);
    if (rowProblem(p) || rowProblem(q)) continue;
    if (!sameIds(p.generated, q.generated)) out.push({ id: p.id, request: "A" });
    else if (p.bSha === q.bSha && !sameIds(p.continuation, q.continuation))
      out.push({ id: p.id, request: "B" });
  }
  return out;
}

// A0's request-B line shows the prompt in the class C1 targets: the final round of request A was a
// drafted round with >= 1 accepted draft, so the spec shadow held a tail (tail distance 1 = eligible).
export const eligibleInA0 = (r) => r.prevRound === "drafted" && r.prevNAcc >= 1;

/**
 * Pre-scoring filter (round-7 structural rule). Every prompt is either scorable or excluded once, with
 * ONE reason, and every exclusion is either attributable to the shell arm C or not:
 *   NOT C (VOID territory): A0's or A1's row is invalid (rowProblem: a failed request, or too few
 *     tokens) - a control failure, checked FIRST so a common outage is never charged to C; A1's request
 *     A differs from A0's (determinism: gateRefusal returns void-determinism before this matters); A0's
 *     or A1's request-B restore line is missing or has tail_dist != 1; a benign arm answered a
 *     different B; A0's line shows no eligible tail (no accepted draft in the final round) and C did
 *     not restore from a tail.
 *   C (checked only when A0's and A1's rows are valid and A1 reproduced A0's request A): C's row is
 *     invalid by the same rowProblem rule (no row, a failed or unparsed request A or B, too few
 *     tokens); C's request A differs from A0's; C's request B differs from A0's; on a prompt A0 shows as
 *     eligible, C's request-B restore line is missing, has tail_dist != 1, or is not a strict tail
 *     restore. Each records the request that failed (A or B) and why.
 * All C exclusions count against ONE shared slack (prompts - minPrompts). preVerdict: more than the
 * slack = `shell-diverged` (STOP); up to the slack they are counted drops.
 * restoresByArm: {arm: census restore rows (in log order)}.
 */
export function engagementAndDrops(records, restoresByArm, { shell, minPrompts }) {
  const byArm = new Map(records.map((r) => [r.arm, r]));
  const a0 = byArm.get("A0");
  if (!a0) throw new Error("missing arm record arm-A0.json");
  const matched = {};
  for (const a of ["A0", "A1", shell].filter((x) => byArm.has(x))) {
    matched[a] = matchRestores(byArm.get(a), restoresByArm[a] ?? [], a0);
  }
  const rowOf = (arm, id) => byArm.get(arm)?.prompts.find((p) => p.id === id);
  const drop = new Map();
  const droppedCounts = {};
  const cExclusions = [];
  const cReasons = {};
  const nonCReasons = {};
  const exclude = (id, byC, request, reason, detail = "") => {
    const label = byC ? `${shell}: ${reason}` : reason;
    drop.set(id, detail ? `${label} (${detail})` : label);
    droppedCounts[label] = (droppedCounts[label] ?? 0) + 1;
    if (byC) {
      cExclusions.push({ id, request, reason, ...(detail ? { detail } : {}) });
      cReasons[reason] = (cReasons[reason] ?? 0) + 1;
    } else {
      nonCReasons[reason] = (nonCReasons[reason] ?? 0) + 1;
    }
  };
  // the reason a constrained arm's request-B line cannot be used, or null
  const lineProblem = (m) =>
    !m
      ? "no request-B restore line"
      : m.tailDist !== 1
        ? `restore line tail_dist=${m.tailDist}`
        : null;
  let tailRestores = 0;
  let eligible = 0;
  let controlExcluded = 0;
  for (const p of a0.prompts) {
    const id = p.id;
    const a0Line = matched.A0[id];
    if (a0Line && a0Line.tailDist === 1 && eligibleInA0(a0Line)) eligible += 1;
    // the controls first: a prompt without a valid A0 AND A1 row is a control failure, never C's
    const a0Bad = rowProblem(p);
    if (a0Bad) {
      controlExcluded += 1;
      exclude(id, false, null, `A0: invalid control row (${a0Bad.reason})`, a0Bad.detail);
      continue;
    }
    const a1Row = rowOf("A1", id);
    const a1Bad = rowProblem(a1Row);
    if (a1Bad) {
      controlExcluded += 1;
      exclude(id, false, null, `A1: invalid control row (${a1Bad.reason})`, a1Bad.detail);
      continue;
    }
    if (!sameIds(a1Row.generated, p.generated)) {
      exclude(id, false, null, "A1: request A differs from A0's (determinism)");
      continue;
    }
    // C, row level: whatever went wrong on C's own requests (same validity rule as score)
    const cRow = rowOf(shell, id);
    const cBad = rowProblem(cRow);
    if (cBad) {
      exclude(id, true, cBad.request, cBad.reason, cBad.detail);
      continue;
    }
    if (!sameIds(cRow.generated, p.generated)) {
      exclude(id, true, "A", "request A output differs from A0's");
      continue;
    }
    if (cRow.bSha !== p.bSha) {
      exclude(id, true, "B", "request B differs from A0's");
      continue;
    }
    // not C: the reference arms' own lines, and the benign arms' B
    const a0Problem = lineProblem(a0Line);
    if (a0Problem) {
      exclude(id, false, null, `A0: ${a0Problem}`);
      continue;
    }
    const a1Problem = lineProblem(matched.A1?.[id]);
    if (a1Problem) {
      exclude(id, false, null, `A1: ${a1Problem}`);
      continue;
    }
    const otherB = records.find((r) => {
      const row = rowOf(r.arm, id);
      return r.arm !== shell && row?.ok && row.bSha !== p.bSha;
    });
    if (otherB) {
      exclude(id, false, null, `${otherB.arm}: request B differs from A0's`);
      continue;
    }
    // C's request-B line: a strict tail restore scores; otherwise C's fault iff A0 shows a tail
    const mc = matched[shell]?.[id];
    if (mc && mc.tailDist === 1 && isTailRestore(mc)) {
      tailRestores += 1;
      continue;
    }
    if (eligibleInA0(a0Line)) {
      const lp = lineProblem(mc);
      exclude(
        id,
        true,
        "B",
        lp ? lp.replace(/tail_dist=.*$/, "tail_dist != 1") : "not a strict tail restore",
        lp ??
          `${mc.chosenOrigin}/${mc.outcome}/${mc.reason}${mc.tailAvailable ? "" : ", no tail written"}`,
      );
    } else {
      exclude(
        id,
        false,
        null,
        "A0: no eligible tail (final round without an accepted draft)",
        `prev_round=${a0Line.prevRound} prev_n_acc=${a0Line.prevNAcc}`,
      );
    }
  }
  const slack = Math.max(0, a0.prompts.length - minPrompts);
  return {
    engaged: tailRestores >= minPrompts,
    tailRestores,
    eligible,
    slack,
    cExcluded: cExclusions.length,
    cReasons,
    cExclusions,
    nonCExcluded: drop.size - cExclusions.length,
    nonCReasons,
    controlExcluded,
    drop,
    droppedCounts,
    droppedPrompts: drop.size,
  };
}

/**
 * The verdict before scoring, from engagementAndDrops' result, or null to score.
 *   - more C-attributable exclusions than the shared slack: `shell-diverged` (STOP), with the
 *     per-reason breakdown and each prompt's failed request;
 *   - fewer than minPrompts scorable strict tail restores, with the C exclusions within the slack: the
 *     shortfall is not C's: `insufficient-control` (VOID) when invalid A0/A1 rows alone exceed the
 *     slack, else `not-engaged:no-eligible-prompts` (VOID).
 */
export function preVerdict(pre, { shell, minPrompts }) {
  const breakdown = (o) =>
    Object.entries(o)
      .map(([k, v]) => `${k}: ${v}`)
      .join("; ");
  if (pre.cExcluded > pre.slack) {
    const which = pre.cExclusions.map((e) => `${e.id}@${e.request}`).join(", ");
    return {
      verdict: "shell-diverged",
      voidReason: `${pre.cExcluded} prompt(s) excluded because of arm ${shell} (shared slack ${pre.slack}): ${breakdown(pre.cReasons)} [${which}]`,
    };
  }
  if (pre.engaged) return null;
  if (pre.controlExcluded > pre.slack) {
    return {
      verdict: "insufficient-control",
      voidReason: `${pre.controlExcluded} prompt(s) without a valid A0/A1 control row (slack ${pre.slack}): ${breakdown(pre.nonCReasons)}; the gate did not answer`,
    };
  }
  return {
    verdict: "not-engaged:no-eligible-prompts",
    voidReason: `only ${pre.tailRestores} scorable strict tail restores in ${shell} (< ${minPrompts}); ${pre.nonCExcluded} exclusion(s) not attributable to ${shell} (${breakdown(pre.nonCReasons)}), ${pre.cExcluded} attributable to ${shell} within the slack: the gate did not answer`,
  };
}

/**
 * Everything before scoring, in the coordinator's order (round 8). Throws (exit 1) unless the records of
 * A0, A1, the shell arm and every benign arm all exist. Then:
 *   1. preScoreChecks (CUDA STOP, flags VOID, PLE STOP, port VOID);
 *   2. determinism: A1 differing from A0 on ANY prompt with valid rows (request A, or B's continuation)
 *      = `void-determinism` (VOID; the plan's fallback is one rerun with speculation off) - BEFORE any C
 *      attribution, so spec-on nondeterminism is never charged to C;
 *   3. preVerdict (C exclusions vs the shared slack, engagement).
 * Returns {pre, refused}: refused is a verdict object, or null to score with pre.drop.
 */
export function gateRefusal(records, censusByArm, restoresByArm, { shell, benign, minPrompts }) {
  const have = new Set(records.map((r) => r.arm));
  const missing = ["A0", "A1", shell, ...benign].filter((a) => !have.has(a));
  if (missing.length) {
    throw new Error(
      `refusing to score: missing arm record(s) ${missing.map((a) => `arm-${a}.json`).join(", ")} (A0, A1, ${shell} and ${benign.join(", ")} are all required)`,
    );
  }
  const pre = engagementAndDrops(records, restoresByArm, { shell, minPrompts });
  const checks = preScoreChecks(records, censusByArm, { shell });
  if (checks) return { pre, refused: checks };
  const nd = determinismFailures(records);
  if (nd.length) {
    return {
      pre,
      refused: {
        verdict: "void-determinism",
        voidReason: `A1 differed from A0 on ${nd.length} prompt(s) (${nd.map((f) => `${f.id}@${f.request}`).join(", ")}): spec-on greedy is not deterministic; rerun once with speculation off in all arms (merged plan section 4 step 4)`,
      },
    };
  }
  return { pre, refused: preVerdict(pre, { shell, minPrompts }) };
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
    return rowProblem(row) ? null : row.continuation; // the same validity rule as engagement
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
      !rowProblem(a0RowA) &&
      !rowProblem(a1RowA) &&
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
        .filter((r) => !rowProblem(r))
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
    verdict === "cuda-errors" ||
    verdict === "ple-hist" ||
    verdict === "shell-diverged"
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
 * Checks on the arms' logs before any scoring. censusByArm: {arm: raw census state (feedFiles: the
 * log + its <LogStem>.flags and <LogStem>.port sidecars)}. Returns a verdict object, or null to proceed
 * to engagement and scoring. In order:
 *   - any CUDA error line in any arm's log: `cuda-errors` (STOP: stops the window);
 *   - an arm whose RECORDED flags differ from its required set (C: REQUIRED_FLAGS.gateTail, others:
 *     REQUIRED_FLAGS.gateOff/gateA5/gateA3), or with no flags record: `mislaunched` (VOID). Decided only
 *     from the launcher's record, never from the lever's own output. Every required set has
 *     PLE_HIST_REWIND=1 + PLE_HIST_LOG=1, so past this point each arm's PLE repair was armed and logged;
 *   - a `[ple-hist] reset` at pos > 0 in any arm's log (an unrepaired rewind): `ple-hist` (STOP);
 *   - an arm whose <LogStem>.port record is missing or does not show the launched PID as the only
 *     listener on the port: `mislaunched` (VOID: requests may have reached another server);
 *   - an arm's log with no `[ple-hist] set` line: `ple-hist` (STOP).
 * HTTP/connection and parse failures of C's requests are C-attributable exclusions (engagementAndDrops),
 * counted against the shared slack.
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
  const perArm = (fn) =>
    records
      .map((r) => {
        const why = fn(censusByArm[r.arm], r.arm);
        return why ? `${r.arm}: ${why}` : null;
      })
      .filter(Boolean);
  const mislaunched = perArm((c, arm) =>
    flagsMismatch(c?.flags ?? null, requiredFlagsOfArm(arm, shell), c?.flagsProblem ?? null),
  );
  if (mislaunched.length) {
    return {
      verdict: "mislaunched",
      voidReason: `recorded launch flags do not match the step-4 arms: ${mislaunched.join("; ")}`,
    };
  }
  const resets = perArm((c) =>
    c.ple.resetsAfterPos0 > 0 ? `[ple-hist] reset at pos > 0 = ${c.ple.resetsAfterPos0}` : null,
  );
  if (resets.length) {
    return {
      verdict: "ple-hist",
      voidReason: `unrepaired PLE-history rewinds: ${resets.join("; ")}`,
    };
  }
  const ports = perArm((c) => portProblemOf(c));
  if (ports.length) {
    return {
      verdict: "mislaunched",
      voidReason: `port ownership not verified: ${ports.join("; ")}`,
    };
  }
  const unarmed = perArm((c) => (c.ple.sets > 0 ? null : "no [ple-hist] set line"));
  if (unarmed.length) {
    return {
      verdict: "ple-hist",
      voidReason: `PLE-history repair recorded as on but never logged a set: ${unarmed.join("; ")}`,
    };
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
    // the log plus its <LogStem>.flags (recorded launch flags) and <LogStem>.port (port owner) sidecars
    censusByArm[r.arm] = await feedFiles([o.armLogs[r.arm]]);
    restoresByArm[r.arm] = censusByArm[r.arm].v2.restores;
  }
  // throws (exit 1) unless A0, A1, the shell arm and every benign arm have records
  const { pre, refused } = gateRefusal(records, censusByArm, restoresByArm, {
    shell: o.shell,
    benign: o.benign,
    minPrompts: o.minPrompts,
  });
  const horizon = records[0]?.horizon ?? o.horizon;
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
      eligibleInA0: pre.eligible,
      required: o.minPrompts,
      engaged: pre.engaged,
      slack: pre.slack,
      shellExcluded: pre.cExcluded,
      shellReasons: pre.cReasons,
      shellExclusions: pre.cExclusions,
      otherExcluded: pre.nonCExcluded,
      otherReasons: pre.nonCReasons,
      controlExcluded: pre.controlExcluded,
    },
    droppedPrompts: pre.droppedPrompts,
    droppedCounts: pre.droppedCounts,
    ...result,
  };
  fs.writeFileSync(file, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(
    `engagement: ${pre.tailRestores} scorable strict tail restores in ${o.shell} (need ${o.minPrompts}); ${pre.eligible} prompts eligible in A0\n`,
  );
  process.stdout.write(
    `excluded because of ${o.shell}: ${pre.cExcluded} of slack ${pre.slack} ${JSON.stringify(pre.cReasons)}\n`,
  );
  for (const e of pre.cExclusions) {
    process.stdout.write(
      `  ${e.id} request ${e.request}: ${e.reason}${e.detail ? ` (${e.detail})` : ""}\n`,
    );
  }
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
