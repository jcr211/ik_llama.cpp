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
//      below the 32-checkpoint cap. Every arm sends the same request sequence. Draft state (round 12):
//      every step-4 arm runs the MTP drafter ONLY (launcher -MtpOnly, drafters=mtp), so there is no
//      server-lifetime ngram-mod table (fed every request's output, pre-empting MTP, reset after three
//      low-acceptance rounds) to drift between A0 and C after a benign B divergence; MTP drafts come
//      from the model's own head, per request. Tail eligibility still comes from MTP drafted rounds.
//      The round-11 safeguards stay as a backstop: "no strict tail restore" is charged only with tail
//      evidence or C's OWN eligible round (rule 9), and a C request-A difference (which would now need
//      a greedy batch-shape difference within one MTP config) is charged to C;
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
// Arms (one server launch each through launch-stateos-tail-8099.ps1 with its own -LogStem, ALL with
// -DivLog -MtpOnly (round 12); the launcher sets LONGSPEAR_PLE_HIST_REWIND=1 + LONGSPEAR_PLE_HIST_LOG=1):
//   A0 flag off (RUN FIRST: the other arms read its record), A1 flag-off repeat (determinism control),
//   C = -Tail, benign A5 (-ExtraArgs '-no-fmoe -no-fug') and A3 (-ExtraArgs '-fa 0') flag off.
// Runbook: after each launch run check-stateos-port-8099.ps1 ONCE, before any traffic; never re-run it
// after the arm's traffic (its .port mtime must precede the record's startedAt).
//
// Arm records: `run` needs --log-stem <the server's launcher -LogStem>; the record carries it with url,
// horizon, nFirst, receipt, bRequest (request B's extra body, {ignore_eos: true} in every arm),
// startedAt/finishedAt and per row: sha256 (prompt content), generated (request A ids), continuation
// (request B ids), bSha, bFinish, eosAt (B ended on EOS before the horizon), failedAt/errorKind.
//
// Scoring needs every arm's record and server log (--arm-log ARM=path). Each log's <LogStem>.flags
// sidecar (launcher) records the effective flags, spec=on|off, drafters= and logstem=; <LogStem>.port
// (written by check-stateos-port-8099.ps1) and <LogStem>.pid are the only port evidence.
//
// ONE validity rule (rowProblem) for engagement and scoring: a row is invalid when a request failed,
// request A returned < 2 tokens, or request B returned fewer tokens than the horizon - including an
// EOS before the horizon (row.eosAt), for every arm, because the v2 lib cannot express a genuine end.
// A restore line binds to a row by request order AND content (n_past == the forced index, cache-window
// marked token == g_A0[G-2], prompt-window marked token == X); a C row whose request B reached the
// server but failed is bound with A0's row fields. Benign arms reach the same B from their own cache,
// so their restore lines are not constrained.
//
// VERDICT PRECEDENCE (the first rule that fires decides; tested pairwise in the tools test):
//   0. refusal - error, exit 1:
//      - checkRecords: exactly one record per arm A0, A1, C, A5, A3 and no other; one horizon, nFirst
//        and receipt; bRequest == {ignore_eos: true} in every record; A0's prompt ids AND sha256 values
//        in every record; url exactly http://127.0.0.1:8099; each record's logStem == its --arm-log
//        stem == that .flags logstem=; distinct log stems; a PID shared by two arms only if their
//        [port check, finishedAt] intervals do not overlap (sequential reuse is fine); each record's
//        startedAt after its own port check (the .port mtime);
//      - receipt binding: A0's prompt ids and sha256 values == the receipt's prompts in order (all of
//        them, or a leading prefix for a --smoke --limit run);
//      - preregistered parameters (PREREG): the default receipt (by sha256), 24 prompts, horizon 256,
//        n-first 64, min-prompts 20, n-min 6, shell C, benign A5,A3 (and drafters=mtp, enforced as rule
//        2). Otherwise "non-preregistered parameters", or with --smoke a run labelled SMOKE that exits
//        4 (never 0 or 2);
//   1. CUDA error lines in any arm's log - `cuda-errors` STOP;
//   2. flags/spec/drafters mismatch - `mislaunched` VOID: recorded flags differ from the arm's required
//      set (C: DIV_LOG + TAIL_SNAPSHOT; others DIV_LOG only; all PLE_HIST_REWIND + PLE_HIST_LOG;
//      ExtraArgs A5 '-no-fmoe -no-fug', A3 '-fa 0', others none), no flags record, the arms' spec
//      states mixed/unknown, or any arm not drafters=mtp (drafters=none in the spec-off fallback);
//   3. a `[ple-hist] reset` at pos > 0 in any arm - `ple-hist` STOP;
//   4. a .port record missing or not (ok, port 8099, exactly one listener == .pid) - `mislaunched` VOID;
//   5. no `[ple-hist] set` line in an arm - `ple-hist` STOP;
//   6. every arm spec=off (the plan's fallback) - `spec-off-fallback` VOID, "spec-off fallback:
//      determinism only; C1 not testable", with determinism PASS/FAIL (rule 7's comparison); never C
//      exclusions, never C1 scoring (no drafted rounds = no tail; C1 stays off by default);
//   7. A1 differs from A0 (request A where both As succeeded; B continuation where both B requests
//      succeeded on the same B, at any length) - `void-determinism` VOID;
//   8. invalid A0/A1 rows (control failures, never charged to C) > slack - `insufficient-control` VOID;
//   9. C-attributable exclusions > the ONE shared slack (prompts - min-prompts, 24 - 20 = 4) -
//      `shell-diverged` STOP. C reasons (only where A0's and A1's rows are valid and A1 reproduced A0's
//      request A): C's row invalid; C's request A differs from A0's; C's B differs; C's request-B line
//      shows TAIL EVIDENCE (a tail written/available or chosen, or a tail outcome) without a strict tail
//      restore - whatever the final-round markers say (round 12); on a prompt A0's line shows as
//      eligible (prev_round=drafted, prev_n_acc >= 1): C's request-B line missing or not at
//      tail_dist=1, or - when C's OWN line also shows an eligible round - not a strict tail restore
//      (chosen_origin=tail, outcome=restored, reason=tail). Each is reported with prompt, request (A
//      or B) and reason; up to the slack = counted drops;
//  10. fewer than --min-prompts scorable strict tail restores (the shortfall is not C's: A0/A1 lines,
//      A0 showing no eligible tail, C's own round not eligible and no tail evidence, a benign arm's
//      different B) - `not-engaged:no-eligible-prompts` VOID. gate.json `partial` (and stdout) still
//      reports the v2 score of the strict tail restores that did occur, labelled "partial, not a
//      verdict" (round 12);
//  11. score (v2 rule): compatible-at-horizon PASS; shellWorse STOP; insufficient-sample VOID.
// Exit codes: PASS 0, STOP 2, VOID 3 (the gate did not answer; not a C1 kill), refusal 1, SMOKE 4.
//
// Usage:
//   node tools/stateos-tail-gate.mjs run --arm A0 --log-stem <stem> --out <dir>
//        [--url http://127.0.0.1:8099] [--receipt <v2 receipt.json>] [--n-first 64] [--horizon 256]
//   node tools/stateos-tail-gate.mjs run --arm C --log-stem <stem> --out <dir> [--reference <dir>/arm-A0.json] ...
//   node tools/stateos-tail-gate.mjs score --out <dir> --arm-log A0=<log> --arm-log A1=<log>
//        --arm-log C=<log> --arm-log A5=<log> --arm-log A3=<log> [--lib <noise-floor-lib.mjs>] [--smoke]
//   (the preregistered values are the defaults; --limit, other horizons etc. only make a --smoke run)
// The API key comes from LONGSPEAR_API_KEY, else from the --api-key line of
// D:/AI/llama-swap/config.yaml; it is never printed or written.

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  MODEL_PORT,
  REQUIRED_FLAGS,
  feedFiles,
  flagsMismatch,
  portProblemOf,
} from "./stateos-div-census.mjs";

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
    logStem: undefined,
    lib: DEFAULT_LIB,
    nFirst: 64,
    horizon: 256,
    limit: undefined,
    shell: "C",
    benign: ["A5", "A3"],
    minPrompts: 20,
    nMin: 6,
    smoke: false,
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
    else if (flag === "--log-stem") o.logStem = value();
    else if (flag === "--lib") o.lib = value();
    else if (flag === "--n-first") o.nFirst = Number(value());
    else if (flag === "--horizon") o.horizon = Number(value());
    else if (flag === "--limit") o.limit = Number(value());
    else if (flag === "--shell") o.shell = value();
    else if (flag === "--benign") o.benign = value().split(",").filter(Boolean);
    else if (flag === "--min-prompts") o.minPrompts = Number(value());
    else if (flag === "--n-min") o.nMin = Number(value());
    else if (flag === "--smoke") o.smoke = true;
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
  if (o.cmd === "run" && !/^[A-Za-z0-9._-]+$/.test(o.logStem ?? ""))
    throw new Error("--log-stem <the server's launcher -LogStem> is required");
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

// Request B in EVERY arm (coordinator ruling, round 10): ignore_eos so the continuation reaches the
// horizon. The server bans only llama_token_eos(model) (server-context.cpp: logit_bias[eos] = -inf),
// so another end-of-generation token can still end B early: rowProblem's EOS/short rule stays as the
// backstop.
export const B_REQUEST = Object.freeze({ ignore_eos: true });

// every record's url, exactly (host AND port; round 11)
export const MODEL_URL = `http://127.0.0.1:${MODEL_PORT}`;

// The preregistered step-4 parameters (round 11, coordinator ruling). Anything else is refused as a
// verdict (exit 1, "non-preregistered parameters"), or runs only with --smoke, labelled SMOKE, exit 4
// (never 0 or 2).
// every step-4 arm: the MTP drafter only (round 12; enforced from the .flags records, rule 2)
export const STEP4_DRAFTERS = "mtp";

export const PREREG = Object.freeze({
  // documented here and recorded in gate.json; enforced from the launch records as rule 2 (a
  // mismatch is VOID mislaunched, not a refusal)
  drafters: STEP4_DRAFTERS,
  receipt: DEFAULT_RECEIPT,
  receiptSha256: "7acd327043cb01c860f403380d3df0a50306e2ee1c06a818549ee26eb44a7cf3",
  prompts: 24,
  horizon: 256,
  nFirst: 64,
  minPrompts: 20,
  nMin: 6,
  shell: "C",
  benign: Object.freeze(["A5", "A3"]),
});

const sha256Hex = (text) => crypto.createHash("sha256").update(text).digest("hex");

/**
 * Prompt content binding (round 11): A0's rows (ids and sha256, in order) must equal the receipt's
 * inputsEcho.prompts, or a leading prefix of them (a --limit smoke run; the full set is enforced by
 * preregProblems); checkRecords binds every other arm to A0's. receiptText: the receipt file's
 * contents (null = unreadable). Returns the problems (empty = bound).
 */
export function receiptProblems(records, receiptText) {
  const a0 = records.find((r) => r.arm === "A0");
  if (!a0) return ["no A0 record"];
  if (receiptText == null) return [`receipt ${a0.receipt} unreadable`];
  let prompts;
  try {
    prompts = JSON.parse(receiptText)?.inputsEcho?.prompts;
  } catch {
    prompts = null;
  }
  if (!Array.isArray(prompts)) return [`receipt ${a0.receipt} has no inputsEcho.prompts`];
  // A0's rows must be the receipt's prompts in order: all of them for the gate (preregProblems then
  // requires the full 24), or a leading prefix for a --smoke --limit run (round 12, Sol nit)
  const want = JSON.stringify(prompts.slice(0, a0.prompts.length).map((p) => [p.id, p.sha256]));
  const got = JSON.stringify(a0.prompts.map((p) => [p.id, p.sha256 ?? null]));
  return a0.prompts.length > 0 && a0.prompts.length <= prompts.length && want === got
    ? []
    : [
        `A0's prompt ids/sha256 are not the receipt's prompts in order (${a0.prompts.length} rows vs ${prompts.length})`,
      ];
}

/** Differences from PREREG (empty = the preregistered gate). receiptText as in receiptProblems. */
export function preregProblems(records, { minPrompts, nMin, shell, benign }, receiptText) {
  const a0 = records.find((r) => r.arm === "A0");
  const bad = [];
  const want = (name, got, need) => {
    if (JSON.stringify(got) !== JSON.stringify(need))
      bad.push(`${name}=${JSON.stringify(got)} (preregistered ${JSON.stringify(need)})`);
  };
  want("receipt", a0?.receipt, PREREG.receipt);
  want("receipt sha256", receiptText == null ? null : sha256Hex(receiptText), PREREG.receiptSha256);
  want("prompts", a0?.prompts.length, PREREG.prompts);
  want("horizon", a0?.horizon, PREREG.horizon);
  want("n-first", a0?.nFirst, PREREG.nFirst);
  want("min-prompts", minPrompts, PREREG.minPrompts);
  want("n-min", nMin, PREREG.nMin);
  want("shell", shell, PREREG.shell);
  want("benign", benign, PREREG.benign);
  return bad;
}

// The spec-off fallback (every arm -SpecOff) checks determinism only: the tail comes from the spec
// shadow and needs a drafted round, so with speculation off C1 cannot engage (coordinator ruling).
export const SPEC_OFF_LABEL = "spec-off fallback: determinism only; C1 not testable";

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
async function complete(o, key, prompt, n, extra = {}) {
  let resp;
  try {
    resp = await post(o.url, key, "/v1/completions", {
      ...GREEDY,
      prompt,
      max_tokens: n,
      n_predict: n,
      ...extra,
    });
  } catch (e) {
    e.kind = "http";
    throw e;
  }
  try {
    // finish: "length" (max_tokens reached) or "stop" (EOS / a stop string)
    return { ...tokensOf(resp), finish: resp?.choices?.[0]?.finish_reason ?? null };
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
    // the server's launcher -LogStem: score ties this record to that log and its .flags record
    logStem: o.logStem,
    nFirst: o.nFirst,
    horizon: o.horizon,
    // request B's extra body, the same in every arm (checkRecords requires it identical)
    bRequest: B_REQUEST,
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
      const b = await complete(o, key, forced.tokens, o.horizon, B_REQUEST);
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
        bFinish: b.finish,
        // a genuine end before the horizon (EOS): recorded; rowProblem counts the row invalid because
        // the v2 lib scores a short vector as a divergence / an equal short pair as the full horizon
        ...(b.ids.length < o.horizon && b.finish === "stop" ? { eosAt: b.ids.length } : {}),
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

// request A must return >= 2 tokens (B replaces the second-to-last one); request B must reach the
// record's horizon (rowProblem)
export const MIN_A_TOKENS = 2;

const requireHorizon = (horizon) => {
  if (!Number.isInteger(horizon) || horizon < 1)
    throw new Error(`the arm records carry no valid horizon (${horizon})`);
  return horizon;
};

/**
 * THE validity rule for an arm's row, shared by engagementAndDrops, the determinism check and score
 * (so score never rejects a continuation that engagement counted as valid). null when valid, else
 * {request: "A" | "B", reason, detail}. Invalid: a failed request; a request A with fewer than 2 tokens;
 * a request-B continuation shorter than the requested horizon. A continuation that ended on EOS before
 * the horizon (row.eosAt) is invalid too, for EVERY arm: the v2 lib cannot express a genuine end (it
 * scores a shorter vector as a divergence at its length, and an equal short pair as surviving to the
 * full horizon).
 */
export function rowProblem(row, horizon) {
  requireHorizon(horizon);
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
  if (nB < horizon) {
    return {
      request: "B",
      reason:
        row.eosAt != null
          ? "request B ended on EOS before the horizon"
          : "request B returned fewer tokens than the horizon",
      detail: `${nB} of ${horizon} tokens`,
    };
  }
  return null;
}

// request A alone is usable (for the determinism control) when it succeeded with >= 2 tokens, whatever
// happened to request B
const requestAUsable = (row) =>
  Boolean(row) &&
  (row.ok || row.failedAt === "B") &&
  Array.isArray(row.generated) &&
  row.generated.length >= MIN_A_TOKENS;

/**
 * Determinism control (merged plan section 4 step 4): A1 must reproduce A0 - request A's ids on every
 * prompt where both request As succeeded (even when a request B then failed), and B's continuation
 * where both B requests succeeded on the same B, at any length. Returns the failing prompts [{id, request}]. Checked
 * BEFORE any C attribution or slack: spec-on nondeterminism would otherwise show up as C differences.
 */
export function determinismFailures(records) {
  const a0 = records.find((r) => r.arm === "A0");
  const a1 = records.find((r) => r.arm === "A1");
  if (!a0 || !a1) return [];
  const out = [];
  for (const p of a0.prompts) {
    const q = a1.prompts.find((x) => x.id === p.id);
    if (requestAUsable(p) && requestAUsable(q) && !sameIds(p.generated, q.generated)) {
      out.push({ id: p.id, request: "A" });
    } else if (
      // both B requests succeeded on the same B: any difference, at ANY length (an EOS-ended or short
      // continuation included), is nondeterminism (round 11, Opus N1)
      p?.ok &&
      q?.ok &&
      p.bSha === q.bSha &&
      !sameIds(p.continuation, q.continuation)
    ) {
      out.push({ id: p.id, request: "B" });
    }
  }
  return out;
}

// A0's request-B line shows the prompt in the class C1 targets: the final round of request A was a
// drafted round with >= 1 accepted draft, so the spec shadow held a tail (tail distance 1 = eligible).
export const eligibleInA0 = (r) => r.prevRound === "drafted" && r.prevNAcc >= 1;

// C's request-B line shows a tail existed for this restore: written/available before it, chosen, or
// named in the outcome/reason (e.g. restore-failed after a tail choice, restored:xcheck-flag-off)
export const tailEvidence = (r) =>
  Boolean(
    r &&
      (r.tailAvailable ||
        r.chosenOrigin === "tail" ||
        /tail/.test(String(r.outcome ?? "")) ||
        /tail/.test(String(r.reason ?? ""))),
  );

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
  const horizon = requireHorizon(a0.horizon);
  const rowProblemOf = (row) => rowProblem(row, horizon);
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
    const a0Bad = rowProblemOf(p);
    if (a0Bad) {
      controlExcluded += 1;
      exclude(id, false, null, `A0: invalid control row (${a0Bad.reason})`, a0Bad.detail);
      continue;
    }
    const a1Row = rowOf("A1", id);
    const a1Bad = rowProblemOf(a1Row);
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
    const cBad = rowProblemOf(cRow);
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
    // C's request-B line: a strict tail restore scores. Otherwise:
    //  - TAIL EVIDENCE FIRST (round 12, Sol): if C's line shows a tail was written/available or chosen
    //    (tailAvailable, chosen_origin=tail, or a tail outcome/reason) and the restore was not strict,
    //    that is C's fault whatever C's or A0's final-round marker says (a root-only final round can
    //    keep an older eligible shadow tail);
    //  - where A0 shows a tail: a missing or misplaced line is C's fault; "no strict tail restore" is
    //    C's fault when C's OWN line shows an eligible final round (round 11, Opus B1); otherwise C's
    //    round legitimately held no accepted draft (no tail to write) - not C-attributable.
    const mc = matched[shell]?.[id];
    if (mc && mc.tailDist === 1 && isTailRestore(mc)) {
      tailRestores += 1;
      continue;
    }
    if (mc && mc.tailDist === 1 && tailEvidence(mc)) {
      exclude(
        id,
        true,
        "B",
        "not a strict tail restore",
        `${mc.chosenOrigin}/${mc.outcome}/${mc.reason}, tail available=${mc.tailAvailable}`,
      );
      continue;
    }
    if (eligibleInA0(a0Line)) {
      const lp = lineProblem(mc);
      if (lp) {
        exclude(id, true, "B", lp.replace(/tail_dist=.*$/, "tail_dist != 1"), lp);
      } else if (eligibleInA0(mc)) {
        exclude(
          id,
          true,
          "B",
          "not a strict tail restore",
          `${mc.chosenOrigin}/${mc.outcome}/${mc.reason}${mc.tailAvailable ? "" : ", no tail written"}`,
        );
      } else {
        exclude(
          id,
          false,
          null,
          `${shell}: final round not eligible (drafting differed from A0's)`,
          `prev_round=${mc.prevRound} prev_n_acc=${mc.prevNAcc}`,
        );
      }
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
 * The verdict before scoring, from engagementAndDrops' result, or null to score. In order:
 *   - more invalid A0/A1 control rows than the slack: `insufficient-control` (VOID);
 *   - more C-attributable exclusions than the shared slack: `shell-diverged` (STOP), with the
 *     per-reason breakdown and each prompt's failed request;
 *   - fewer than minPrompts scorable strict tail restores: the shortfall is not C's,
 *     `not-engaged:no-eligible-prompts` (VOID).
 */
export function preVerdict(pre, { shell, minPrompts }) {
  const breakdown = (o) =>
    Object.entries(o)
      .map(([k, v]) => `${k}: ${v}`)
      .join("; ");
  // control failures first: with more invalid A0/A1 rows than the slack the gate cannot attribute
  // anything to C (round 9: VOID wins over shell-diverged)
  if (pre.controlExcluded > pre.slack) {
    return {
      verdict: "insufficient-control",
      voidReason: `${pre.controlExcluded} prompt(s) without a valid A0/A1 control row (slack ${pre.slack}): ${breakdown(pre.nonCReasons)}; the gate did not answer`,
    };
  }
  if (pre.cExcluded > pre.slack) {
    const which = pre.cExclusions.map((e) => `${e.id}@${e.request}`).join(", ");
    return {
      verdict: "shell-diverged",
      voidReason: `${pre.cExcluded} prompt(s) excluded because of arm ${shell} (shared slack ${pre.slack}): ${breakdown(pre.cReasons)} [${which}]`,
    };
  }
  if (pre.engaged) return null;
  return {
    verdict: "not-engaged:no-eligible-prompts",
    voidReason: `only ${pre.tailRestores} scorable strict tail restores in ${shell} (< ${minPrompts}); ${pre.nonCExcluded} exclusion(s) not attributable to ${shell} (${breakdown(pre.nonCReasons)}), ${pre.cExcluded} attributable to ${shell} within the slack: the gate did not answer`,
  };
}

/**
 * Rule 0 (round 9): the five arm records must be ONE consistent run, else refuse to score (throws;
 * exit 1). Exactly one record per arm A0, A1, the shell arm and every benign arm, and no other record
 * (a stale `arm-A0-attempt1.json` is refused, never picked); the same horizon, nFirst and receipt in
 * every record; request B's body (bRequest) equal to B_REQUEST in every record; every record has A0's ordered prompt-id list; every record's url is on the model port;
 * every record names the launcher -LogStem of its server, equal to the logstem= of the arm log's
 * .flags record (censusByArm[arm].logStem), which ties every arm, benign ones included, to its own log.
 */
export function checkRecords(records, censusByArm, { shell, benign }) {
  const expected = ["A0", "A1", shell, ...benign];
  const bad = [];
  const count = {};
  for (const r of records) count[r.arm] = (count[r.arm] ?? 0) + 1;
  const missing = expected.filter((a) => !count[a]);
  if (missing.length)
    bad.push(`missing arm record(s) ${missing.map((a) => `arm-${a}.json`).join(", ")}`);
  const dup = expected.filter((a) => count[a] > 1);
  if (dup.length) bad.push(`more than one record for arm(s) ${dup.join(", ")}`);
  const extra = Object.keys(count).filter((a) => !expected.includes(a));
  if (extra.length) bad.push(`unexpected arm record(s) ${extra.join(", ")}`);
  if (!bad.length) {
    const a0 = records.find((r) => r.arm === "A0");
    const ids = JSON.stringify(a0.prompts.map((p) => p.id));
    // prompt CONTENT: every row carries the prompt's sha256, equal to A0's (round 11, Sol bug 2)
    const shas = JSON.stringify(a0.prompts.map((p) => p.sha256 ?? null));
    if (a0.prompts.some((p) => typeof p.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(p.sha256)))
      bad.push("A0: a prompt row has no sha256");
    // distinct launches (round 11, Opus N4): distinct log stems and PIDs, and each arm's traffic
    // started after its own port check (the .port mtime)
    const stems = records.map((r) => r.logStem);
    if (new Set(stems).size !== stems.length)
      bad.push(`log stems are not distinct across arms (${stems.join(", ")})`);
    // a shared PID is refused only when the two arms' launch-to-last-request intervals ([port check,
    // finishedAt]) overlap or cannot be established; sequential PID reuse by Windows is fine (round 12)
    const span = (r) => {
      const from = Date.parse(censusByArm?.[r.arm]?.portCheckedAt ?? "");
      const to = Date.parse(r.finishedAt ?? "");
      return Number.isFinite(from) && Number.isFinite(to) ? [from, to] : null;
    };
    for (let i = 0; i < records.length; i += 1) {
      for (let j = i + 1; j < records.length; j += 1) {
        const [ri, rj] = [records[i], records[j]];
        const [pi, pj] = [censusByArm?.[ri.arm]?.pid, censusByArm?.[rj.arm]?.pid];
        if (pi == null || pi !== pj) continue;
        const [si, sj] = [span(ri), span(rj)];
        if (!si || !sj || (si[0] <= sj[1] && sj[0] <= si[1]))
          bad.push(
            `${ri.arm} and ${rj.arm} share launched PID ${pi} over overlapping (or unknown) launch-to-last-request intervals`,
          );
      }
    }
    for (const r of records) {
      if (JSON.stringify(r.prompts.map((p) => p.sha256 ?? null)) !== shas)
        bad.push(`${r.arm}: prompt sha256 values differ from A0's`);
      const checkedAt = censusByArm?.[r.arm]?.portCheckedAt ?? null;
      if (checkedAt !== null && !(Date.parse(r.startedAt) > Date.parse(checkedAt)))
        bad.push(
          `${r.arm}: record startedAt ${JSON.stringify(r.startedAt ?? null)} is not after its port check ${checkedAt}`,
        );
      for (const k of ["horizon", "nFirst", "receipt"]) {
        if (r[k] !== a0[k])
          bad.push(`${r.arm}: ${k}=${JSON.stringify(r[k])} (A0: ${JSON.stringify(a0[k])})`);
      }
      // request B's protocol (ignore_eos) must be the same in every arm, and the current one
      if (JSON.stringify(r.bRequest ?? null) !== JSON.stringify(B_REQUEST))
        bad.push(
          `${r.arm}: request-B body ${JSON.stringify(r.bRequest ?? null)} (every arm needs ${JSON.stringify(B_REQUEST)})`,
        );
      if (JSON.stringify(r.prompts.map((p) => p.id)) !== ids)
        bad.push(
          `${r.arm}: prompt ids differ from A0's (${r.prompts.length} vs ${a0.prompts.length})`,
        );
      if (r.url !== MODEL_URL)
        bad.push(`${r.arm}: url ${JSON.stringify(r.url)} (need exactly ${MODEL_URL})`);
      const logStem = censusByArm?.[r.arm]?.logStem ?? null;
      if (!r.logStem || r.logStem !== logStem)
        bad.push(
          `${r.arm}: record logStem ${JSON.stringify(r.logStem ?? null)} != its log's .flags logstem ${JSON.stringify(logStem)}`,
        );
    }
    if (!Number.isInteger(a0.horizon) || a0.horizon < 1)
      bad.push(`A0: horizon ${a0.horizon} invalid`);
  }
  if (bad.length) {
    throw new Error(
      `refusing to score: the arm records are not one consistent run: ${bad.join("; ")}`,
    );
  }
}

/**
 * Everything before scoring, in the full precedence order (round 9; see the header):
 *   0. checkRecords (throws: exit 1);
 *   1-5. preScoreChecks (CUDA STOP, flags/spec VOID, PLE reset STOP, port VOID, PLE set STOP);
 *   6. spec-off fallback (every arm spec=off): `spec-off-fallback` (VOID) with determinism PASS/FAIL;
 *   7. determinism: A1 differing from A0 = `void-determinism` (VOID) - BEFORE any C attribution, so
 *      spec-on nondeterminism is never charged to C;
 *   8-10. preVerdict (insufficient-control VOID, C slack STOP, not-engaged VOID).
 * Returns {pre, refused, specOff}: refused is a verdict object, or null to score with pre.drop; specOff
 * is true when every arm was launched with -SpecOff (the plan's spec-off fallback run).
 */
export function gateRefusal(records, censusByArm, restoresByArm, { shell, benign, minPrompts }) {
  checkRecords(records, censusByArm, { shell, benign });
  const specOff = records.every((r) => censusByArm?.[r.arm]?.flags?.spec === "off");
  const pre = engagementAndDrops(records, restoresByArm, { shell, minPrompts });
  const checks = preScoreChecks(records, censusByArm, { shell });
  if (checks) return { pre, refused: checks, specOff };
  const nd = determinismFailures(records);
  const ndList = nd.map((f) => `${f.id}@${f.request}`).join(", ");
  // the spec-off fallback answers ONE question, determinism, and stops here: never C exclusions,
  // never C1 scoring (round 11)
  if (specOff) {
    return {
      pre,
      specOff,
      refused: {
        verdict: "spec-off-fallback",
        determinism: nd.length ? "FAIL" : "PASS",
        determinismFailures: nd,
        voidReason: `${SPEC_OFF_LABEL}; determinism ${nd.length ? `FAIL: A1 differed from A0 on ${nd.length} prompt(s) (${ndList}) even with speculation off` : "PASS: A1 reproduced A0 on every prompt"}`,
      },
    };
  }
  if (nd.length) {
    return {
      pre,
      specOff,
      refused: {
        verdict: "void-determinism",
        voidReason: `A1 differed from A0 on ${nd.length} prompt(s) (${ndList}): spec-on greedy is not deterministic; rerun once with speculation off in all arms (launcher -SpecOff; merged plan section 4 step 4)`,
      },
    };
  }
  return { pre, refused: preVerdict(pre, { shell, minPrompts }), specOff };
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
    return rowProblem(row, horizon) ? null : row.continuation; // the same validity rule as engagement
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
      !rowProblem(a0RowA, horizon) &&
      !rowProblem(a1RowA, horizon) &&
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
        .filter((r) => !rowProblem(r, horizon))
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

/**
 * Round 12 (Opus R1): when not-engaged VOID fires, the strict tail restores that DID occur are still
 * scored with the v2 classifier (min-prompts = their count) and reported, labelled "partial, not a
 * verdict", so a C whose restores are visibly corrupt is not read as a neutral VOID. Never changes the
 * verdict.
 */
export function partialScore(records, lib, pre, { shell, benign, horizon, nMin }) {
  const s = score(records, lib, {
    shell,
    benign,
    horizon,
    minPrompts: Math.max(1, pre.tailRestores),
    nMin,
    drop: pre.drop,
  });
  const signs = { earlier: 0, tie: 0, later: 0 };
  for (const r of s.rows) signs[r.sign] += 1;
  return {
    label: "partial, not a verdict",
    strictTailRestores: pre.tailRestores,
    scoredRows: s.rows.length,
    signs,
    shellEarliest: s.rows.filter((r) => r.sEarliest && Math.min(r.shell, r.worstBenign) < horizon)
      .length,
    classifier: s.verdict,
    classifierReason: s.rule?.reason ?? null,
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
  // one speculation state for all arms: all 'on' (the standard run) or all 'off' (the spec-off
  // fallback); mixed or unknown = mislaunched
  const specs = records.map((r) => `${r.arm}=${censusByArm[r.arm]?.flags?.spec ?? "unknown"}`);
  const specValues = new Set(records.map((r) => censusByArm[r.arm]?.flags?.spec ?? "unknown"));
  if (specValues.size !== 1 || !(specValues.has("on") || specValues.has("off"))) {
    return {
      verdict: "mislaunched",
      voidReason: `the arms' recorded speculation states differ or are unknown (${specs.join(", ")}): every arm must be spec=on, or every arm spec=off (the fallback)`,
    };
  }
  // drafters (round 12): every step-4 arm runs the MTP drafter ONLY (launcher -MtpOnly), no ngram-mod
  // table to drift across requests; the spec-off fallback has none
  const needDrafters = specValues.has("off") ? "none" : STEP4_DRAFTERS;
  const wrongDrafters = records
    .map((r) => [r.arm, censusByArm[r.arm]?.flags?.drafters ?? "unknown"])
    .filter(([, d]) => d !== needDrafters);
  if (wrongDrafters.length) {
    return {
      verdict: "mislaunched",
      voidReason: `step 4 needs drafters=${needDrafters} in every arm (launcher ${needDrafters === "none" ? "-SpecOff" : "-MtpOnly"}): ${wrongDrafters.map(([a, d]) => `${a}=${d}`).join(", ")}`,
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
  // every arm-*.json in --out is read; a file not named exactly arm-<its arm>.json (e.g. a stale
  // arm-A0-attempt1.json) is refused, never silently picked or skipped
  const records = fs
    .readdirSync(o.out)
    .filter((f) => /^arm-.+\.json$/.test(f))
    .map((f) => {
      const r = JSON.parse(fs.readFileSync(path.join(o.out, f), "utf8"));
      if (f !== `arm-${r.arm}.json`)
        throw new Error(
          `refusing to score: ${f} holds arm ${JSON.stringify(r.arm)}; move stale records out of --out`,
        );
      return r;
    });
  for (const r of records) {
    if (!o.armLogs[r.arm])
      throw new Error(`--arm-log ${r.arm}=<server log> is required for every arm`);
    // the record's -LogStem must name the log it is scored against
    const stem = path
      .basename(o.armLogs[r.arm])
      .replace(/\.err\.log$/i, "")
      .replace(/\.log$/i, "");
    if (r.logStem !== stem)
      throw new Error(
        `refusing to score: arm ${r.arm}'s record was run against log stem ${JSON.stringify(r.logStem ?? null)}, but --arm-log gives ${stem}`,
      );
  }
  const censusByArm = {};
  const restoresByArm = {};
  for (const r of records) {
    // the log plus its <LogStem>.flags (recorded launch flags) and <LogStem>.port (port owner) sidecars
    censusByArm[r.arm] = await feedFiles([o.armLogs[r.arm]]);
    restoresByArm[r.arm] = censusByArm[r.arm].v2.restores;
  }
  // throws (exit 1) unless the records are one consistent run (checkRecords)
  const { pre, refused, specOff } = gateRefusal(records, censusByArm, restoresByArm, {
    shell: o.shell,
    benign: o.benign,
    minPrompts: o.minPrompts,
  });
  // rule 0, continued: prompt content bound to the receipt; preregistered parameters, else refuse
  // (exit 1) - or, with --smoke, run labelled SMOKE with exit 4 (never 0 or 2)
  const a0 = records.find((r) => r.arm === "A0");
  const receiptText = fs.existsSync(a0.receipt) ? fs.readFileSync(a0.receipt, "utf8") : null;
  const unbound = receiptProblems(records, receiptText);
  if (unbound.length) throw new Error(`refusing to score: ${unbound.join("; ")}`);
  const offPrereg = preregProblems(records, o, receiptText);
  if (offPrereg.length && !o.smoke) {
    throw new Error(
      `refusing to score: non-preregistered parameters: ${offPrereg.join("; ")} (use --smoke for a labelled smoke run)`,
    );
  }
  const horizon = records[0].horizon; // equal in every record (checkRecords)
  const runLabel = `${o.smoke ? "SMOKE (not the gate) - " : ""}${specOff ? SPEC_OFF_LABEL : "standard (spec on)"}`;
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
  // not-engaged VOID: still score the strict tail restores that did occur, report-only
  const partial =
    result.verdict === "not-engaged:no-eligible-prompts" && pre.tailRestores > 0
      ? partialScore(records, lib, pre, { shell: o.shell, benign: o.benign, horizon, nMin: o.nMin })
      : null;
  const status = o.smoke ? "SMOKE" : gateStatus(result.verdict);
  const file = path.join(o.out, "gate.json");
  const report = {
    scoredAt: new Date().toISOString(),
    lib: o.lib,
    armLogs: o.armLogs,
    run: runLabel,
    // every parameter the verdict depends on, and whether it is the preregistered gate
    parameters: {
      preregistered: offPrereg.length === 0 && !o.smoke,
      offPreregistration: offPrereg,
      receipt: a0.receipt,
      receiptSha256: receiptText == null ? null : sha256Hex(receiptText),
      prompts: a0.prompts.length,
      horizon,
      nFirst: a0.nFirst,
      minPrompts: o.minPrompts,
      nMin: o.nMin,
      shell: o.shell,
      benign: o.benign,
      bRequest: a0.bRequest,
      drafters: Object.fromEntries(
        records.map((r) => [r.arm, censusByArm[r.arm]?.flags?.drafters ?? null]),
      ),
      specOff,
    },
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
    ...(partial ? { partial } : {}),
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
    `verdict [${runLabel}]: ${result.verdict}${result.voidReason ? ` (${result.voidReason})` : ""}\n`,
  );
  process.stdout.write(`rule: ${result.rule.outcome} - ${result.rule.reason ?? ""}\n`);
  if (result.determinism) process.stdout.write(`determinism: ${result.determinism}\n`);
  if (partial) {
    process.stdout.write(
      `PARTIAL, NOT A VERDICT: ${partial.strictTailRestores} strict tail restores, ${partial.scoredRows} scored; signs ${JSON.stringify(partial.signs)}; ${partial.shellEarliest} shell-earliest events; v2 classifier on them: ${partial.classifier}${partial.classifierReason ? ` (${partial.classifierReason})` : ""}\n`,
    );
  }
  process.stdout.write(
    `status: ${status}${status === "VOID" ? " (the gate did not answer; not a C1 kill)" : status === "SMOKE" ? " (not the preregistered gate; no verdict)" : ""}\n`,
  );
  process.stdout.write(`wrote ${file}\n`);
  process.exitCode = status === "SMOKE" ? 4 : status === "PASS" ? 0 : status === "VOID" ? 3 : 2;
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
