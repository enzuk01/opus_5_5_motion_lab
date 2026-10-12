#!/usr/bin/env python3
"""Measure the Liquid Chrome score against its brief.

usage: verify_score.py <dir>            (liquid-chrome.wav, stem_*.f32, cues.json)
       verify_score.py <dir> decoded.f32le  (also check the MP4's audio, decoded by ffmpeg to interleaved f32le)

Reports loudness, true peak, DC, the spectrum split, the 200 Hz to 6 kHz phone band,
mono fold-down, each hit's level over the bed and where its first waveform peak lands
relative to frame + 1.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

SR, FPS, LAND = 48000, 60, 1


def read_wav24(path):
    b = Path(path).read_bytes()
    assert b[:4] == b'RIFF' and b[8:12] == b'WAVE'
    i, fmt, data = 12, None, None
    while i < len(b):
        cid, size = b[i:i + 4], int.from_bytes(b[i + 4:i + 8], 'little')
        if cid == b'fmt ':
            fmt = b[i + 8:i + 8 + size]
        elif cid == b'data':
            data = b[i + 8:i + 8 + size]
        i += 8 + size + (size & 1)
    ch, sr, bits = int.from_bytes(fmt[2:4], 'little'), int.from_bytes(fmt[4:8], 'little'), int.from_bytes(fmt[14:16], 'little')
    assert sr == SR and bits == 24, (sr, bits)
    raw = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
    v = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
    v = np.where(v >= 1 << 23, v - (1 << 24), v)
    return (v.reshape(-1, ch).T / 8388607.0), ch, sr, bits


def read_f32(path):
    x = np.fromfile(path, np.float32)
    return x.reshape(2, -1).astype(np.float64)


def kweight(x):
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1, -2, 1], [1, -1.99004745483398, 0.99007225036621]
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=-1), axis=-1)


def lufs(x):
    z = kweight(np.concatenate([x, x], axis=-1))          # the loop, played twice
    block, hop = int(0.4 * SR), int(0.1 * SR)
    ms = np.array([np.sum(z[:, s:s + block] ** 2) / block for s in range(0, z.shape[1] - block + 1, hop)])
    L = -0.691 + 10 * np.log10(ms + 1e-30)
    g = ms[L > -70]
    rel = -0.691 + 10 * np.log10(g.mean()) - 10
    g = ms[L > -70][(-0.691 + 10 * np.log10(g + 1e-30)) > rel]
    return -0.691 + 10 * np.log10(g.mean())


def true_peak_db(x):
    up = signal.resample_poly(np.concatenate([x, x, x], axis=-1), 4, 1, axis=-1)
    n = x.shape[-1] * 4
    return 20 * np.log10(np.max(np.abs(up[:, n:2 * n])))


def bandpass(x, lo, hi):
    sos = signal.butter(4, [lo, hi], 'bandpass', fs=SR, output='sos')
    y = signal.sosfilt(sos, np.concatenate([x, x], axis=-1), axis=-1)   # circular warm-up
    return y[:, x.shape[-1]:]


def db(v):
    return 10 * np.log10(v + 1e-30)


def windows(cues):
    t = lambda f: (f + LAND) / FPS
    w = {
        'reverse swell (66-84)': (t(84) - 0.12, t(84)),
        'snap (84)': (t(84), t(84) + 0.12),
        'pop 1 (108)': (t(108), t(108) + 0.06),
        'pop 2 (112)': (t(112), t(112) + 0.06),
        'air lift (112-126)': (t(126) - 0.15, t(126)),
        'form swell peak (171)': (t(171) - 0.06, t(171) + 0.05),
        'glint shimmer (198-246)': (t(198), t(246)),
        'let-go click (288)': (t(288), t(288) + 0.06),
        'falling swish (290-318)': (t(290) + 0.08, t(290) + 0.3),
        'whoosh (318-337)': (t(337) - 0.15, t(337)),
        'shut (337)': (t(337), t(337) + 0.15),
        'tick 1 (350)': (t(350), t(350) + 0.03),
        'tick 2 (359)': (t(359), t(359) + 0.03),
    }
    impulses = {'snap (84)': 84, 'pop 1 (108)': 108, 'pop 2 (112)': 112, 'let-go click (288)': 288,
                'shut (337)': 337, 'tick 1 (350)': 350, 'tick 2 (359)': 359}
    return w, impulses


def first_peak_ms(x, f, search=0.02, thresh=0.5):
    """first local maximum of |x| above thresh * window max, in ms after frame + 1"""
    t0 = (f + LAND) / FPS
    a, b = int(round(t0 * SR)) - int(0.002 * SR), int(round((t0 + search) * SR))
    seg = np.abs(np.take(x, np.arange(a, b), mode='wrap'))
    m = seg.max()
    for i in range(1, len(seg) - 1):
        if seg[i] >= thresh * m and seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]:
            return (a + i) / SR * 1000 - t0 * 1000, seg[i] / m
    return float('nan'), 0


def report(d, decoded=None):
    d = Path(d)
    mix, ch, sr, bits = read_wav24(d / 'liquid-chrome.wav')
    stems = {k: read_f32(d / f'stem_{k}.f32') for k in ('bed', 'hits', 'kick')}
    cues = json.loads((d / 'cues.json').read_text())
    n = mix.shape[1]
    print(f'format: {ch} ch, {sr} Hz, {bits}-bit, {n} samples ({n / SR:.3f} s)')
    print(f'integrated loudness: {lufs(mix):.2f} LUFS, true peak: {true_peak_db(mix):.2f} dBTP')
    print(f'DC: L {mix[0].mean():+.2e}, R {mix[1].mean():+.2e}')

    mono = mix.mean(axis=0)
    spec = np.abs(np.fft.rfft(mono * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / SR)
    tot = spec.sum()
    frac = lambda lo, hi: spec[(f >= lo) & (f < hi)].sum() / tot * 100
    med = f[np.searchsorted(np.cumsum(spec), tot / 2)]
    print(f'spectrum: <60 Hz {frac(0, 60):.1f}%, 60-150 Hz {frac(60, 150):.1f}%, 150 Hz-6 kHz {frac(150, 6000):.1f}%, '
          f'6-8 kHz {frac(6000, 8000):.2f}%, >8 kHz {frac(8000, SR / 2):.4f}% ({db(frac(8000, SR / 2) / 100):.1f} dB), median {med:.0f} Hz')
    phone = bandpass(mix, 200, 6000)
    print(f'phone band (200 Hz-6 kHz): {lufs(phone):.2f} LUFS')
    m2 = np.vstack([mono, mono])
    print(f'mono fold-down: {lufs(m2) - lufs(mix):+.2f} LU vs stereo, L/R correlation {np.corrcoef(mix[0], mix[1])[0, 1]:.3f}')

    print('\nhits over the bed, phone band, K-weighted (target 3 to 6 dB):')
    hp, bp = [kweight(bandpass(stems[k], 200, 6000)) for k in ('hits', 'bed')]
    w, impulses = windows(cues)
    for name, (a, b) in w.items():
        idx = np.arange(int(a * SR), int(b * SR)) % n
        diff = db(np.mean(hp[:, idx] ** 2)) - db(np.mean(bp[:, idx] ** 2))
        print(f'  {name:26s} {diff:+5.1f} dB')

    print('\nfirst waveform peak after frame + 1 (hit stem):')
    hm = stems['hits'].mean(axis=0)
    for name, fr in impulses.items():
        ms, rel = first_peak_ms(hm, fr)
        print(f'  {name:26s} {ms:+5.2f} ms')
    km = stems['kick'].mean(axis=0)
    print('kick (first peak, largest peak, level 3 ms after frame + 1):')
    for fr in range(0, 360, 60):
        t0 = int(round((fr + LAND) / FPS * SR))
        seg = np.abs(km[t0 - 96:t0 + int(0.02 * SR)])
        first, _ = first_peak_ms(km, fr, thresh=0.3)
        big = (np.argmax(seg) - 96) / SR * 1000
        env = np.abs(signal.hilbert(km[t0 - 480:t0 + 4800]))[480:]
        print(f'  frame {fr:3d}: first {first:+.2f} ms, largest {big:+.2f} ms, 3 ms level {env[int(0.003 * SR)] / env[:int(0.001 * SR)].max() * 100:.0f}%')

    if decoded:
        # ffmpeg -f f32le output is interleaved; the stems are planar
        dec = np.fromfile(decoded, np.float32).reshape(-1, 2).T.astype(np.float64)
        L = min(dec.shape[1], n)
        xc = signal.correlate(dec[0, :L], mix[0, :L], mode='full', method='fft')
        lag = np.argmax(xc) - (L - 1)
        print(f'\ndecoded MP4 audio: {dec.shape[1]} samples, offset {lag / SR * 1000:+.3f} ms vs the WAV')
        dl = np.roll(dec[:, :L], -lag, axis=1) if abs(lag) < SR else dec[:, :L]
        print(f'  loudness {lufs(dl):.2f} LUFS, true peak {true_peak_db(dl):.2f} dBTP')
        resid = dl.mean(axis=0) - (mix[:, :L] - read_f32(Path(d) / 'stem_hits.f32')[:, :L]).mean(axis=0)
        print('  hits in the decoded audio (decoded minus mastered bed), first peak after frame + 1:')
        for name, fr in impulses.items():
            ms, _ = first_peak_ms(resid, fr)
            print(f'    {name:26s} {ms:+5.2f} ms')


if __name__ == '__main__':
    report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
