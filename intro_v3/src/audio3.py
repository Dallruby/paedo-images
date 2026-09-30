import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile
SR = 44100; DUR = 24.5; N = int(SR * DUR)
r = np.random.default_rng(1); tt = np.arange(N) / SR
L = np.zeros(N); R = np.zeros(N)
def T(n): return np.arange(n) / SR
def filt(x, k, f, o=2): return sosfilt(butter(o, f, btype=k, fs=SR, output='sos'), x)
def place(sig, t0, g=1.0, pan=0.0):
    i = int(t0 * SR); sig = sig[:max(0, N - i)]
    gl, gr = g * np.sqrt((1 - pan) / 2), g * np.sqrt((1 + pan) / 2)
    if sig.ndim == 1: L[i:i + len(sig)] += sig * gl; R[i:i + len(sig)] += sig * gr
    else: L[i:i + len(sig)] += sig[:, 0] * gl; R[i:i + len(sig)] += sig[:, 1] * gr
def reverb(x, secs=2.0, decay=3.0, seed=0, wet=0.35):
    rr = np.random.default_rng(seed); n = int(secs * SR)
    ir = filt(rr.normal(size=n) * np.exp(-decay * T(n)), 'lowpass', 5000); ir /= np.sqrt((ir ** 2).sum())
    y = np.zeros(len(x) + n); yc = fftconvolve(x, ir); y[:len(yc)] = yc[:len(y)]
    o = np.zeros(len(x) + n); o[:len(x)] = x * (1 - wet); return o + y * wet
def nz(n): return r.normal(size=n)
def norm(x): return x / (np.abs(x).max() + 1e-9)

# rain: outside (0-9.6) full, inside muffled
def rain(seed):
    rr = np.random.default_rng(seed)
    b = filt(rr.normal(size=N), 'bandpass', [400, 9000]) * 0.25 + filt(rr.normal(size=N), 'highpass', 6000) * 0.08
    d = np.zeros(N); ii = rr.integers(0, N, int(DUR * 900)); d[ii] = rr.normal(size=len(ii)) * 0.8
    return b + filt(d, 'bandpass', [1500, 7000]) * 0.5
rL, rR = rain(10), rain(11)
ext = np.clip(tt / 1.2, 0, 1) * (tt < 9.6) * np.where(tt >= 7.0, 1.15, 1.0)
inn = (tt >= 9.6) * (1 - np.clip((tt - 23.6) / 0.9, 0, 1)) * np.where(tt >= 17.2, 0.55, 1.0)
for x, ch in ((rL, L), (rR, R)):
    ch += x * ext * 0.5 + (filt(x, 'lowpass', 900) * 1.6) * inn * 0.45

def thunder(dur, crack, lp, seed):
    n = int(dur * SR); rr = np.random.default_rng(seed)
    b = filt(np.cumsum(rr.normal(size=n)), 'highpass', 25); b = norm(b)
    k = np.interp(T(n), np.linspace(0, dur, 14), rr.random(14) * 0.7 + 0.3)
    e = np.exp(-T(n) * 4.8 / dur) * k; e[:int(0.08 * SR)] *= np.linspace(0, 1, int(0.08 * SR))
    o = filt(b, 'lowpass', lp) * 3 * e
    if crack:
        nc = int(0.35 * SR); c = rr.normal(size=nc) * np.exp(-T(nc) * 14)
        o[:nc] += (filt(c, 'highpass', 300) * 0.9 + filt(c, 'lowpass', 300) * 1.5) * crack
    return norm(o)
place(norm(reverb(thunder(3.5, 0.2, 450, 1), 2.5, 2.0, 1, 0.4)), 0.95, 0.6, -0.3)
place(norm(reverb(thunder(4.0, 1.0, 900, 2), 3.0, 1.8, 2, 0.35)), 6.98, 0.9)
place(norm(filt(reverb(thunder(2.5, 0.3, 400, 3), 2.0, 2.2, 3, 0.4), 'lowpass', 800)), 11.45, 0.5, 0.4)
place(norm(filt(reverb(thunder(2.5, 0.5, 500, 4), 2.0, 2.2, 4, 0.4), 'lowpass', 900)), 12.5, 0.55, 0.5)
place(norm(filt(reverb(thunder(2.0, 0.3, 400, 5), 2.0, 2.2, 5, 0.4), 'lowpass', 700)), 16.7, 0.4, -0.4)
place(norm(filt(reverb(thunder(3.0, 0.6, 500, 6), 2.5, 2.0, 6, 0.4), 'lowpass', 800)), 22.75, 0.6)

# drone
x = np.zeros(N)
for fq, a in [(41.2, 1.0), (41.6, 0.8), (61.7, 0.45), (82.4, 0.3), (98.0, 0.15)]:
    ph = 2 * np.pi * fq * tt + 0.3 * np.sin(2 * np.pi * 0.13 * tt); x += a * np.sin(ph) + 0.25 * a * np.sin(2 * ph)
x = norm(filt(x, 'lowpass', 300))
e = np.interp(tt, [0, 1, 5.5, 6.9, 7.0, 7.3, 9.6, 12.4, 14.2, 17.2, 19.4, 22.6, 23.8, 24.5], [0, .35, .5, .95, 1, .35, .4, .5, .55, .5, .6, .8, .2, 0])
L += x * e * 0.3; R += x * e * 0.3

def siren(dur):
    n = int(dur * SR); t_ = T(n); fq = 700 + 250 * np.sin(2 * np.pi * 0.55 * t_)
    s = filt(np.sin(2 * np.pi * np.cumsum(fq) / SR), 'lowpass', 1400) * np.sin(np.pi * t_ / dur) ** 1.5
    return norm(reverb(s, 2.5, 1.5, 7, 0.6))
place(siren(4.5), 1.5, 0.07, 0.6)
def car(dur, seed):
    n = int(dur * SR); t_ = T(n); x = filt(np.random.default_rng(seed).normal(size=n), 'bandpass', [300, 4000])
    e = np.exp(-((t_ - dur / 2) / (dur / 5)) ** 2)
    return np.stack([x * e * np.clip(1 - t_ / dur, 0, 1), x * e * np.clip(t_ / dur, 0, 1)], 1)
place(car(2.2, 20), 0.2, 0.25); place(car(2.4, 21)[:, ::-1], 2.4, 0.18)
# riser into the window + impact
n = int(1.6 * SR); t_ = T(n); xr = nz(n); o = np.zeros(n)
for i in range(0, n, 2048):
    u = i / n; fc = 300 + 5000 * u ** 2; o[i:i + 2048] = filt(xr[i:i + 2048], 'bandpass', [fc * .6, fc * 1.4], 1)
place(norm((o * 2 + np.sin(2 * np.pi * np.cumsum(80 + 160 * (t_ / 1.6) ** 2) / SR) * .3) * (t_ / 1.6) ** 2.2), 5.4, 0.35)
def impact(f0=70, dur=2.5):
    n = int(dur * SR); t_ = T(n)
    s = np.sin(2 * np.pi * np.cumsum(f0 * np.exp(-t_ * 1.2) + 28) / SR) * np.exp(-t_ * 1.8)
    return norm(reverb(s + filt(nz(n), 'lowpass', 2000) * np.exp(-t_ * 18) * .5, 2.5, 2.0, 9, 0.3))
place(impact(), 7.0, 0.55)
# interior: clock ticks
def tick():
    n = int(0.03 * SR); return filt(nz(n), 'bandpass', [2000, 6000]) * np.exp(-T(n) * 250)
for k in range(18): place(tick(), 9.8 + k * 0.5, 0.05, 0.5)
# painting reveal: low taiko-like hit + tiger growl rumble
def thud(f0=55, d=0.7):
    n = int(d * SR); t_ = T(n); return np.sin(2 * np.pi * (f0 * np.exp(-t_ * 4) + 30) * t_) * np.exp(-t_ * 6)
place(thud(), 12.4, 0.5)
n = int(1.4 * SR); t_ = T(n)
growl = filt(nz(n), 'bandpass', [60, 260]) * (0.6 + 0.4 * np.sin(2 * np.pi * 18 * t_)) * np.sin(np.pi * t_ / 1.4)
place(norm(reverb(growl, 1.5, 3, 12, 0.3)), 12.55, 0.22)
# boss: leather creak, cigarette handling
def creak(d=0.5):
    n = int(d * SR); t_ = T(n)
    return filt(nz(n), 'bandpass', [180, 900]) * (np.sin(2 * np.pi * 23 * t_) > 0.2) * np.sin(np.pi * t_ / d)
place(creak(), 14.35, 0.12, -0.2)
place(filt(nz(int(0.2 * SR)), 'bandpass', [2000, 7000]) * np.exp(-T(int(0.2 * SR)) * 25), 15.3, 0.12)
place(thud(80, 0.4) * 0.5, 14.2, 0.35)
# lighter
def clink(seed, b=1.0):
    n = int(0.5 * SR); t_ = T(n); rr = np.random.default_rng(seed); s = np.zeros(n)
    for fq in [2400, 3710, 5230, 6890, 8100]: s += np.sin(2 * np.pi * fq * t_) * np.exp(-t_ * (18 + fq / 400)) * rr.random()
    return (s + filt(nz(n), 'highpass', 3000) * np.exp(-t_ * 120) * .8) * b
place(reverb(clink(1), 0.8, 6, 11, 0.2), 17.8, 0.35, 0.2)
n = int(0.14 * SR); place(filt(nz(n), 'bandpass', [1500, 9000]) * np.exp(-T(n) * 30), 18.18, 0.5, 0.2)
n = int(0.6 * SR); place(filt(nz(n), 'lowpass', 900) * np.exp(-T(n) * 5) * (1 - np.exp(-T(n) * 60)) * 1.5, 18.2, 0.4, 0.2)
n = int(0.7 * SR); place(filt(nz(n), 'bandpass', [300, 1500]) * 0.3 * np.sin(np.pi * T(n) / 0.7), 18.25, 0.25, 0.2)
place(reverb(clink(2, 1.3), 0.8, 6, 12, 0.2), 18.88, 0.42, 0.2)
# inhale (burn crackle) / exhale "huu"
def burn(d):
    n = int(d * SR); cr = np.zeros(n); ii = r.integers(0, n, int(d * 70)); cr[ii] = r.normal(size=len(ii))
    return (filt(cr, 'bandpass', [1200, 8000]) * 1.5 + filt(nz(n), 'bandpass', [300, 2500]) * .25) * np.sin(np.pi * T(n) / d)
place(burn(1.1), 19.45, 0.35, 0.3)
place(thud(50, 0.6), 19.4, 0.4)
n = int(1.8 * SR); t_ = T(n)
ex = (filt(nz(n), 'bandpass', [250, 3000]) * .6 + filt(nz(n), 'bandpass', [3000, 7000]) * .1) * np.clip(t_ / 0.1, 0, 1) * np.exp(-t_ * 1.6)
place(reverb(ex, 1.2, 4, 13, 0.25), 21.1, 0.55, -0.2)
# final braam
n = int(2.2 * SR); t_ = T(n); xb = np.zeros(n)
for fq, a in [(41.2, 1), (41.5, .8), (61.7, .6), (82.4, .5), (123.5, .2)]: xb += (2 * ((fq * t_) % 1) - 1) * a
xb = filt(xb, 'lowpass', 700) * (1 - np.exp(-t_ * 18)) * np.exp(-t_ * 1.3)
place(norm(reverb(xb, 3.0, 1.5, 21, 0.35)), 22.6, 0.65)
m = np.stack([L, R], 1); m = filt(m.T, 'highpass', 25).T
m = np.tanh(m / np.abs(m).max() * 1.3) * 0.89
m *= np.clip((DUR - tt) / 0.3, 0, 1)[:, None]
wavfile.write('intro3.wav', SR, (m * 32767).astype(np.int16)); print('ok')
