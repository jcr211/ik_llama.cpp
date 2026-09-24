// W-SL1 step 0 (VRAM preflight): one request whose prompt is at least --min tokens (default 131072) and at
// most --max (default 180000; the served context is 196608), then --n 256 decode tokens at the production
// sampler. The prompt is the fork's own C/C++ sources, measured with the server's /tokenize. The API key is
// read from the llama-swap config and never printed. Exit 0 = the request completed; prints prompt/decode
// token counts. Usage: node sl1-long-request.mjs [--url http://127.0.0.1:8099] [--min N] [--max N] [--n N]
import fs from 'node:fs';
import path from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
    if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1]]);
    return acc;
}, []));
const url = args.url ?? 'http://127.0.0.1:8099';
const minTok = Number(args.min ?? 131072);
const maxTok = Number(args.max ?? 180000);
const nPredict = Number(args.n ?? 256);
const srcRoot = args.src ?? 'D:/AI/ik_llama-qwen4exp';

const config = fs.readFileSync('D:/AI/llama-swap/config.yaml', 'utf8');
const key = (config.match(/sk-lm-[A-Za-z0-9]+/) ?? [])[0];
if (!key) {
    console.error('no API key in D:/AI/llama-swap/config.yaml');
    process.exit(2);
}
const headers = { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` };

function sources() {
    const dirs = ['src', 'src/graphs', 'common', 'examples/server', 'ggml/src', 'ggml/src/iqk'];
    const files = [];
    for (const d of dirs) {
        const abs = path.join(srcRoot, d);
        if (!fs.existsSync(abs)) continue;
        for (const f of fs.readdirSync(abs).sort()) {
            if (/\.(c|cpp|h)$/.test(f)) files.push(path.join(abs, f));
        }
    }
    return files;
}

async function countTokens(text) {
    const r = await fetch(`${url}/tokenize`, { method: 'POST', headers, body: JSON.stringify({ content: text }) });
    if (!r.ok) throw new Error(`/tokenize HTTP ${r.status}`);
    const j = await r.json();
    return j.tokens.length;
}

let text = '';
for (const f of sources()) {
    text += `\n// ==== ${path.basename(f)} ====\n` + fs.readFileSync(f, 'utf8');
    if (text.length > maxTok * 5) break;
}
let n = await countTokens(text);
for (let i = 0; i < 6 && (n < minTok || n > maxTok); ++i) {
    if (n < minTok) throw new Error(`sources give only ${n} tokens, need ${minTok}`);
    const target = (minTok + maxTok) / 2;
    text = text.slice(0, Math.floor(text.length * target / n));
    n = await countTokens(text);
}
if (n < minTok || n > maxTok) throw new Error(`could not fit the prompt into [${minTok}, ${maxTok}] tokens (${n})`);
console.log(`[long-request] prompt tokens=${n} n_predict=${nPredict} ${new Date().toISOString()}`);

// streamed, so the response headers arrive before the long prefill ends
const body = {
    prompt: text, n_predict: nPredict, stream: true, cache_prompt: false,
    temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0,
};
const r = await fetch(`${url}/completion`, { method: 'POST', headers, body: JSON.stringify(body) });
if (!r.ok) {
    console.error(`/completion HTTP ${r.status}`);
    process.exit(3);
}
let last = null;
let buf = '';
const dec = new TextDecoder();
for await (const chunk of r.body) {
    buf += dec.decode(chunk, { stream: true });
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (line.startsWith('data:')) {
            try { last = JSON.parse(line.slice(5)); } catch { /* keep the previous chunk */ }
        }
    }
}
const t = last?.timings ?? {};
console.log(`[long-request] done prompt_n=${t.prompt_n ?? '?'} predicted_n=${t.predicted_n ?? '?'} prompt_ms=${t.prompt_ms ?? '?'} predicted_ms=${t.predicted_ms ?? '?'} stop=${last?.stop ?? '?'} ${new Date().toISOString()}`);
process.exit(last?.stop ? 0 : 4);
