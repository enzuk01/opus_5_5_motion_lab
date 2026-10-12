#!/usr/bin/env node
// Render Liquid Chrome to an MP4 with its synthesised score.
//
//   node tools/make-video.mjs [outDir] [--three <path to an unpacked three@0.160.0 package>]
//
// 1. renders the 360 frames through window.renderAt(n / 60) in headless Chromium
// 2. renders liquid-chrome-score.html and saves the WAV plus bed / hit / kick stems
// 3. muxes them with ffmpeg (H.264 + AAC) into liquid-chrome.mp4
// 4. decodes the MP4's audio with ffmpeg for tools/verify_score.py
//
// --three serves the jsdelivr importmap URLs from a local copy, for machines without CDN access.
// Needs playwright (with its Chromium) and ffmpeg on PATH.
import { chromium } from 'playwright';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const threeAt = args.indexOf('--three');
const threeDir = threeAt >= 0 ? path.resolve(args.splice(threeAt, 2)[1]) : null;
const outDir = path.resolve(args[0] || path.join(root, 'renders'));
const frameDir = path.join(outDir, 'frames');
fs.mkdirSync(frameDir, { recursive: true });

const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
async function open(file, query = '') {
  const page = await browser.newPage({ viewport: { width: 1080, height: 1080 } });
  page.on('pageerror', e => console.error('[pageerror]', e.message));
  if (threeDir) {
    await page.route('https://cdn.jsdelivr.net/npm/three@0.160.0/**', r => {
      const f = path.join(threeDir, new URL(r.request().url()).pathname.replace('/npm/three@0.160.0/', ''));
      r.fulfill({ status: 200, body: fs.readFileSync(f), contentType: f.endsWith('.json') ? 'application/json' : 'text/javascript' });
    });
  }
  await page.goto('file://' + path.join(root, file) + query);
  await page.waitForFunction(() => window.__ready || window.__error, null, { timeout: 600000 });
  const err = await page.evaluate(() => window.__error);
  if (err) throw new Error(err);
  return page;
}

// 1. frames
const page = await open('liquid-chrome.html', '?t=0');
for (let n = 0; n < 360; n++) {
  const file = path.join(frameDir, `frame_${String(n).padStart(3, '0')}.png`);
  if (fs.existsSync(file)) continue;
  await page.evaluate(n => window.renderAt(n / 60), n);
  await page.locator('#c').screenshot({ path: file });
  if (n % 30 === 0) console.log(`frame ${n}`);
}

// 2. score
const score = await open('liquid-chrome-score.html');
const r = await score.evaluate(() => {
  const s = window.__score;
  const b64 = u8 => { let t = ''; for (let i = 0; i < u8.length; i += 0x8000) t += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(t); };
  const f32 = ch => { const a = new Float32Array(ch[0].length * 2); a.set(ch[0]); a.set(ch[1], ch[0].length); return b64(new Uint8Array(a.buffer)); };
  return { stats: s.stats, wav: b64(s.wav), cues: s.cues, stems: Object.fromEntries(Object.entries(s.stems).map(([k, v]) => [k, f32(v)])) };
});
fs.writeFileSync(path.join(outDir, 'liquid-chrome.wav'), Buffer.from(r.wav, 'base64'));
for (const [k, v] of Object.entries(r.stems)) fs.writeFileSync(path.join(outDir, `stem_${k}.f32`), Buffer.from(v, 'base64'));
fs.writeFileSync(path.join(outDir, 'cues.json'), JSON.stringify(r.cues));
console.log('score', JSON.stringify({ lufs: r.stats.lufs, truePeakDbtp: r.stats.truePeakDbtp, maxLimiterDb: r.stats.maxLimiterDb }));
await browser.close();

// 3. mux, 4. decode the muxed audio back out
const mp4 = path.join(outDir, 'liquid-chrome.mp4');
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', '60', '-i', path.join(frameDir, 'frame_%03d.png'),
  '-i', path.join(outDir, 'liquid-chrome.wav'), '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p',
  '-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-movflags', '+faststart', '-shortest', mp4], { stdio: 'inherit' });
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', mp4, '-vn', '-f', 'f32le', '-ac', '2', '-ar', '48000',
  path.join(outDir, 'decoded.f32le')], { stdio: 'inherit' });
console.log(`wrote ${mp4}`);
console.log(`check: python3 tools/verify_score.py ${outDir} ${path.join(outDir, 'decoded.f32le')}`);
