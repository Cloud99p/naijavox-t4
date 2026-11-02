// Run the 5 sample WAVs against a deployed NaijaVox endpoint and compute the
// real realtime factor. Nothing here assumes a number: if the endpoint does not
// return timing, this script reports "unmeasured" instead of inventing one.
//
// Usage:
//   node t4-benchmark.cjs <endpointId> [--url]
//     default endpoint form: https://api.runpod.ai/v2/<id>/runsync

const fs = require('fs');
const path = require('path');

const KEY = fs.readFileSync(path.join(__dirname, '.runpod-key'), 'utf8').trim();

const CASES = [
  {
    file: 'samples/pidgin_sample1.wav',
    lang: 'pidgin',
    reference: 'on top di injury her uncle no even carry her go hospital for treatment',
  },
  {
    file: 'samples/pidgin_sample2.wav',
    lang: 'pidgin',
    reference: 'she tell don jazzy for december 2016 say as she be',
  },
  {
    file: 'samples/english_sample1.wav',
    lang: 'english',
    reference: "Did it change plain? Yes. yes. Ok that means he was correct so this is if he's right that",
  },
  {
    file: 'samples/igbo_sample1.wav',
    lang: 'igbo',
    reference: null, // no model-card reference: WER not computable, timing still is
  },
  {
    file: 'samples/yoruba_sample1.wav',
    lang: 'yoruba',
    reference: null,
  },
];

const norm = (s) =>
  s
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s']/gu, ' ')
    .split(/\s+/)
    .filter(Boolean);

function wer(ref, hyp) {
  const r = norm(ref);
  const h = norm(hyp);
  const d = Array.from({ length: r.length + 1 }, () => new Array(h.length + 1).fill(0));
  for (let i = 0; i <= r.length; i++) d[i][0] = i;
  for (let j = 0; j <= h.length; j++) d[0][j] = j;
  for (let i = 1; i <= r.length; i++) {
    for (let j = 1; j <= h.length; j++) {
      d[i][j] = Math.min(
        d[i - 1][j] + 1,
        d[i][j - 1] + 1,
        d[i - 1][j - 1] + (r[i - 1] === h[j - 1] ? 0 : 1),
      );
    }
  }
  return { errors: d[r.length][h.length], refLen: r.length, rate: d[r.length][h.length] / r.length };
}

const endpointId = process.argv[2];
if (!endpointId) {
  console.error('usage: node t4-benchmark.cjs <endpointId>');
  process.exit(2);
}

const base = `https://api.runpod.ai/v2/${endpointId}`;

async function submit(audioB64, language) {
  const t0 = Date.now();
  const res = await fetch(`${base}/runsync`, {
    method: 'POST',
    headers: {
      Authorization: 'Bearer ' + KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      input: { mode: 'benchmark', audio_b64: audioB64, language },
    }),
  });
  const roundTripMs = Date.now() - t0;
  const text = await res.text();
  let body = null;
  try {
    body = JSON.parse(text);
  } catch {
    body = { raw: text };
  }
  return { http: res.status, roundTripMs, body };
}

(async () => {
  const rows = [];
  for (const c of CASES) {
    const p = path.join(__dirname, c.file);
    if (!fs.existsSync(p)) {
      console.log(`SKIP ${path.basename(c.file)} (missing)`);
      continue;
    }
    const b64 = fs.readFileSync(p).toString('base64');
    console.log(`\n--- ${path.basename(c.file)} (${c.lang})`);

    let r;
    try {
      r = await submit(b64, c.lang);
    } catch (e) {
      console.log(`  FAIL transport: ${e.message}`);
      rows.push({ file: c.file, ok: false });
      continue;
    }

    if (r.http !== 200) {
      console.log(`  FAIL http ${r.http}: ${JSON.stringify(r.body).slice(0, 300)}`);
      rows.push({ file: c.file, ok: false });
      continue;
    }

    const out = r.body.output || r.body;
    const measured = out.measured === true && typeof out.gpu_seconds === 'number';
    if (!measured) {
      // Deliberately loud: a missing timing field is the exact failure that
      // would let us quote an invented realtime factor downstream.
      console.log('  UNMEASURED — endpoint returned no timing; refusing to compute RTF');
      console.log('  ' + JSON.stringify(out).slice(0, 300));
      rows.push({ file: c.file, ok: false });
      continue;
    }

    const text_ = out.text || '';
    const audioSec = out.audio_seconds;
    const gpuSec = out.gpu_seconds;
    const rtf = audioSec > 0 ? audioSec / gpuSec : null;

    console.log(`  text       : ${text_.slice(0, 90)}${text_.length > 90 ? '…' : ''}`);
    console.log(`  audio_sec  : ${audioSec.toFixed(2)}`);
    console.log(`  gpu_sec    : ${gpuSec.toFixed(2)}`);
    console.log(`  RTF        : ${rtf ? rtf.toFixed(2) + 'x realtime' : 'n/a'}`);
    console.log(`  round-trip : ${(r.roundTripMs / 1000).toFixed(2)}s`);

    let werLine = 'n/a';
    if (c.reference) {
      const w = wer(c.reference, text_);
      werLine = `${(w.rate * 100).toFixed(1)}% (${w.errors}/${w.refLen} words)`;
      console.log(`  WER        : ${werLine}`);
    }
    if (out.model_load_seconds) console.log(`  cold load  : ${out.model_load_seconds}s`);

    rows.push({ file: c.file, ok: true, audioSec, gpuSec, rtf, roundTripMs: r.roundTripMs });
  }

  const ok = rows.filter((r) => r.ok);
  console.log('\n================ SUMMARY ================');
  if (!ok.length) {
    console.log('No measured results. Nothing to report — do not quote an RTF.');
    process.exit(1);
  }
  const totalAudio = ok.reduce((a, r) => a + r.audioSec, 0);
  const totalGpu = ok.reduce((a, r) => a + r.gpuSec, 0);
  console.log(`measured runs        : ${ok.length}/${rows.length}`);
  console.log(`total audio          : ${totalAudio.toFixed(2)}s`);
  console.log(`total gpu            : ${totalGpu.toFixed(2)}s`);
  console.log(`aggregate RTF        : ${(totalAudio / totalGpu).toFixed(2)}x realtime`);
  console.log(`mean per-file RTF    : ${(ok.reduce((a, r) => a + (r.rtf || 0), 0) / ok.length).toFixed(2)}x`);
  console.log(`mean round-trip      : ${(ok.reduce((a, r) => a + r.roundTripMs, 0) / ok.length / 1000).toFixed(2)}s`);
})();
