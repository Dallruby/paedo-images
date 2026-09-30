"""Procedural SFX for the PAEDO intro. Writes intro.wav (stereo 44.1k)."""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR = 44100
DUR = 14.0
N = int(SR * DUR)
r = np.random.default_rng(1)
L = np.zeros(N); R = np.zeros(N)

def t_(n):
    return np.arange(n) / SR

def filt(x, kind, f, order=2):
    sos = butter(order, f, btype=kind, fs=SR, output='sos')
    return sosfilt(sos, x)

def env_adsr(n, a, d, s_lvl, rel):
    e = np.ones(n) * s_lvl
    na, nd, nr = int(a * SR), int(d * SR), int(rel * SR)
    e[:na] = np.linspace(0, 1, na) if na else e[:na]
    e[na:na + nd] = np.linspace(1, s_lvl, len(e[na:na + nd]))
    if nr: e[-nr:] *= np.linspace(1, 0, nr)
    return e

def place(sig, t0, gain=1.0, pan=0.0):
    i = int(t0 * SR)
    if i >= N: return
    sig = sig[:N - i]
    gl = gain * np.sqrt((1 - pan) / 2); gr = gain * np.sqrt((1 + pan) / 2)
    if sig.ndim == 1:
        L[i:i + len(sig)] += sig * gl; R[i:i + len(sig)] += sig * gr
    else:
        L[i:i + len(sig)] += sig[:, 0] * gl; R[i:i + len(sig)] += sig[:, 1] * gr

def reverb(x, secs=2.0, decay=3.0, seed=0, wet=0.35):
    rr = np.random.default_rng(seed)
    n = int(secs * SR)
    ir = rr.normal(size=n) * np.exp(-decay * t_(n))
    ir = filt(ir, 'lowpass', 5000)
    ir /= np.sqrt(np.sum(ir ** 2))
    y = np.zeros(len(x) + n); yc = fftconvolve(x, ir); y[:len(yc)] = yc[:len(y)]
    out = np.zeros(len(x) + n); out[:len(x)] += x * (1 - wet)
    return out + y * wet

def noise(n):
    return r.normal(size=n)

def brown(n):
    b = np.cumsum(r.normal(size=n)); b = filt(b, 'highpass', 20); return b / (np.abs(b).max() + 1e-9)

# ---------------------------------------------------------------- rain
def rain_bed(n, seed):
    rr = np.random.default_rng(seed)
    base = filt(rr.normal(size=n), 'bandpass', [400, 9000]) * 0.25
    hiss = filt(rr.normal(size=n), 'highpass', 6000) * 0.08
    drops = np.zeros(n)
    idx = rr.integers(0, n, int(n / SR * 900))
    drops[idx] = rr.normal(size=len(idx)) * 0.8
    drops = filt(drops, 'bandpass', [1500, 7000])
    return base + hiss + drops * 0.5

rainL = rain_bed(N, 10); rainR = rain_bed(N, 11)
tt = t_(N)
ext_env = np.clip(tt / 1.2, 0, 1) * (tt < 7.47)
int_env = (tt >= 7.47) * (1 - np.clip((tt - 13.2) / 0.8, 0, 1))
rain_int_L = filt(rainL, 'lowpass', 900) * 1.6 + filt(filt(rainL, 'bandpass', [2000, 5000]), 'lowpass', 4000) * 0.3
rain_int_R = filt(rainR, 'lowpass', 900) * 1.6 + filt(filt(rainR, 'bandpass', [2000, 5000]), 'lowpass', 4000) * 0.3
# closeup: quieter
int_env = int_env * np.where(tt >= 10.45, 0.6, 1.0)
L += rainL * ext_env * 0.5 + rain_int_L * int_env * 0.45
R += rainR * ext_env * 0.5 + rain_int_R * int_env * 0.45

# ---------------------------------------------------------------- thunder
def thunder(dur, crack=0.0, lp=600, seed=0):
    n = int(dur * SR)
    rr = np.random.default_rng(seed)
    b = np.cumsum(rr.normal(size=n)); b = filt(b, 'highpass', 25); b /= np.abs(b).max()
    rumble = filt(b, 'lowpass', lp) * 3
    # rolling amplitude
    k = np.interp(t_(n), np.linspace(0, dur, 14), rr.random(14) * 0.7 + 0.3)
    e = np.exp(-t_(n) * 1.6 / dur * 3) * k
    e[:int(0.08 * SR)] *= np.linspace(0, 1, int(0.08 * SR))
    out = rumble * e
    if crack > 0:
        nc = int(0.35 * SR)
        c = rr.normal(size=nc) * np.exp(-t_(nc) * 14)
        c = filt(c, 'highpass', 300) * 0.9 + filt(c, 'lowpass', 300) * 1.5
        out[:nc] += c * crack
    return out / (np.abs(out).max() + 1e-9)

th1 = reverb(thunder(3.2, 0.0, 400, 1), 2.5, 2.0, 1, 0.4)
place(th1, 0.9, 0.55, -0.3)
th2 = reverb(thunder(4.0, 1.0, 900, 2), 3.0, 1.8, 2, 0.35)
place(th2, 7.43, 0.9, 0.0)
th3 = filt(reverb(thunder(2.5, 0.3, 350, 3), 2.0, 2.2, 3, 0.4), 'lowpass', 700)
place(th3 / np.abs(th3).max(), 12.72, 0.55, 0.3)

# ---------------------------------------------------------------- drone (dark noir pad)
def drone():
    x = np.zeros(N)
    for f, a in [(41.2, 1.0), (41.6, 0.8), (61.7, 0.45), (82.4, 0.3), (98.0, 0.15)]:
        ph = 2 * np.pi * f * tt + 0.3 * np.sin(2 * np.pi * 0.13 * tt)
        x += a * np.sin(ph) + 0.25 * a * np.sin(2 * ph)
    x = filt(x, 'lowpass', 300)
    e = np.interp(tt, [0, 1.0, 5.0, 7.2, 7.46, 7.6, 10.4, 10.5, 12.9, 13.9, 14], [0, 0.35, 0.45, 0.9, 1.0, 0.25, 0.4, 0.55, 0.7, 0.0, 0])
    return x / np.abs(x).max() * e

dr = drone()
L += dr * 0.28; R += dr * 0.28

# ---------------------------------------------------------------- siren (distant)
def siren(dur):
    n = int(dur * SR); tl = t_(n)
    f = 700 + 250 * np.sin(2 * np.pi * 0.55 * tl)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) + 0.3 * np.sin(2 * ph)
    s = filt(s, 'lowpass', 1400)
    e = np.sin(np.pi * np.clip(tl / dur, 0, 1)) ** 1.5
    return reverb(s * e, 2.5, 1.5, 7, 0.6)

sr_ = siren(3.8)
place(sr_ / np.abs(sr_).max(), 1.5, 0.07, 0.6)

# ---------------------------------------------------------------- car passes (wet tyres)
def car_pass(dur, seed):
    n = int(dur * SR); tl = t_(n)
    x = np.random.default_rng(seed).normal(size=n)
    x = filt(x, 'bandpass', [300, 4000])
    e = np.exp(-((tl - dur / 2) / (dur / 5)) ** 2)
    eng = np.sin(2 * np.pi * (55 + 10 * (tl / dur)) * tl) * 0.4
    return np.stack([x * e * np.clip(1 - tl / dur, 0, 1) + eng * e * 0.3, x * e * np.clip(tl / dur, 0, 1) + eng * e * 0.3], 1)

place(car_pass(2.2, 20), 0.0, 0.22)
place(car_pass(2.4, 21)[:, ::-1], 1.9, 0.18)

# ---------------------------------------------------------------- riser whoosh into the window
def riser(dur):
    n = int(dur * SR); tl = t_(n)
    x = noise(n)
    out = np.zeros(n)
    seg = 2048
    for i in range(0, n, seg):
        u = i / n
        fc = 300 + 5000 * u ** 2
        blk = x[i:i + seg]
        out[i:i + seg] = filt(blk, 'bandpass', [fc * 0.6, fc * 1.4], 1)
    e = (tl / dur) ** 2.2
    tone = np.sin(2 * np.pi * np.cumsum(80 + 160 * (tl / dur) ** 2) / SR) * 0.3
    return (out * 2 + tone) * e

rs = riser(2.45)
place(rs / np.abs(rs).max(), 5.0, 0.35)

# impact
def impact(f0=70, dur=2.5):
    n = int(dur * SR); tl = t_(n)
    f = f0 * np.exp(-tl * 1.2) + 28
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tl * 1.8)
    nz = filt(noise(n), 'lowpass', 2000) * np.exp(-tl * 18) * 0.5
    return reverb(s + nz, 2.5, 2.0, 9, 0.3)

im1 = impact(); place(im1 / np.abs(im1).max(), 7.46, 0.55)

# ---------------------------------------------------------------- interior details
def tick():
    n = int(0.03 * SR); tl = t_(n)
    return filt(noise(n), 'bandpass', [2000, 6000]) * np.exp(-tl * 250)

for k in range(7):
    place(tick(), 8.0 + k * 0.5, 0.05, 0.5)

def metal_clink(seed, bright=1.0):
    n = int(0.5 * SR); tl = t_(n)
    rr = np.random.default_rng(seed)
    s = np.zeros(n)
    for f in [2400, 3710, 5230, 6890, 8100]:
        s += np.sin(2 * np.pi * f * (1 + rr.normal() * 0.01) * tl) * np.exp(-tl * (18 + f / 400)) * rr.random()
    s += filt(noise(n), 'highpass', 3000) * np.exp(-tl * 120) * 0.8
    return s * bright

def flint():
    n = int(0.12 * SR); tl = t_(n)
    s = filt(noise(n), 'bandpass', [1500, 9000]) * (np.exp(-tl * 30) + 0.3 * (np.sin(2 * np.pi * 70 * tl) > 0.6))
    return s

def whoof():
    n = int(0.6 * SR); tl = t_(n)
    s = filt(noise(n), 'lowpass', 900) * np.exp(-tl * 5) * (1 - np.exp(-tl * 60))
    return s * 1.5

def burn(dur):
    n = int(dur * SR); tl = t_(n)
    cr = np.zeros(n)
    idx = r.integers(0, n, int(dur * 70))
    cr[idx] = r.normal(size=len(idx))
    cr = filt(cr, 'bandpass', [1200, 8000]) * 1.5
    breath = filt(noise(n), 'bandpass', [300, 2500]) * 0.25
    e = np.sin(np.pi * np.clip(tl / dur, 0, 1))
    return (cr + breath) * e

def exhale(dur):
    n = int(dur * SR); tl = t_(n)
    s = filt(noise(n), 'bandpass', [250, 3000]) * 0.6 + filt(noise(n), 'bandpass', [3000, 7000]) * 0.1
    e = np.clip(tl / 0.12, 0, 1) * np.exp(-tl * 2.2)
    return s * e

place(reverb(metal_clink(1), 0.8, 6, 11, 0.2), 9.72, 0.30, 0.25)
place(flint(), 9.93, 0.45, 0.25)
place(whoof(), 9.96, 0.35, 0.25)
place(reverb(metal_clink(2, 1.3), 0.8, 6, 12, 0.2), 10.36, 0.38, 0.25)
place(burn(0.9), 10.5, 0.30, 0.35)
place(reverb(exhale(1.3), 1.2, 4, 13, 0.25), 11.65, 0.40, -0.2)

# cut hit into closeup (low heartbeat thud)
def thud():
    n = int(0.6 * SR); tl = t_(n)
    return np.sin(2 * np.pi * (55 * np.exp(-tl * 4) + 30) * tl) * np.exp(-tl * 7)

place(thud(), 10.45, 0.45)

# final braam
def braam(dur):
    n = int(dur * SR); tl = t_(n)
    x = np.zeros(n)
    for f, a in [(41.2, 1.0), (41.5, 0.8), (61.7, 0.6), (82.4, 0.5), (123.5, 0.2)]:
        ph = 2 * np.pi * f * tl
        saw = 2 * ((ph / (2 * np.pi)) % 1) - 1
        x += saw * a
    x = filt(x, 'lowpass', 700)
    e = (1 - np.exp(-tl * 18)) * np.exp(-tl * 1.3)
    return reverb(x * e, 3.0, 1.5, 21, 0.35)

bm = braam(2.0)
place(bm / np.abs(bm).max(), 13.08, 0.6)

# ---------------------------------------------------------------- master
mix = np.stack([L, R], 1)
mix = filt(mix.T, 'highpass', 25).T
# soft limiter
peak = np.abs(mix).max()
mix = mix / peak * 1.3
mix = np.tanh(mix) * 0.89
fo = np.clip((DUR - tt) / 0.25, 0, 1)
mix *= fo[:, None]
wavfile.write('intro.wav', SR, (mix * 32767).astype(np.int16))
print('ok peak', peak)
