"""PAEDO intro v3 - pixel-art cutscene assembled from GPT pixel keyframes. 384x216 native, 24fps."""
import numpy as np, math, subprocess, sys, imageio_ffmpeg
from PIL import Image

W, H = 384, 216
FPS = 24
DUR = 24.5
NF = int(DUR * FPS)
rng = np.random.default_rng(11)
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
BAY = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16
BAYT = np.tile(BAY, (H // 4 + 1, W // 4 + 1))[:H, :W]

def C(*v): return np.array(v, np.float32)
def clamp01(x): return max(0.0, min(1.0, x))
def seg(t, a, b): return clamp01((t - a) / (b - a))
def ease(x): x = clamp01(x); return x * x * (3 - 2 * x)
def lerp(a, b, u): return a + (b - a) * u

IMG = {}
for n in ['1', '2', '3-1', '3-2', '4-1', '4-2', '4-3', '5-1', '5-2', '5-4', '6-1', '6-2', '6-3', '6-4', '6-5']:
    IMG[n] = np.asarray(Image.open(f'n_{n}.png').convert('RGB'), np.float32)

# ------------------------------------------------ palettes + LUTs (final snap to the shot's palette)
EXTRA = np.array([[150, 165, 195], [205, 214, 235], [235, 240, 255], [255, 90, 20], [255, 170, 40], [255, 238, 170], [255, 255, 235],
                  [170, 178, 200], [118, 126, 150], [72, 80, 104], [230, 30, 25], [255, 120, 100],
                  [96, 52, 34], [140, 80, 50], [186, 116, 72], [226, 158, 100], [248, 196, 130], [70, 36, 24]], np.float32)
def make_lut(pal):
    pal = np.concatenate([pal, EXTRA])
    g = np.arange(64) * 4 + 2
    cube = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
    lut = np.zeros(len(cube), np.int32)
    wts = np.array([0.30, 0.59, 0.11], np.float32)
    for i in range(0, len(cube), 8192):
        d = (((cube[i:i + 8192, None, :] - pal[None]) ** 2) * wts).sum(-1)
        lut[i:i + 8192] = d.argmin(1)
    return pal.astype(np.uint8), lut
LUTS = {s: make_lut(np.load(f'pal_{s}.npy')) for s in ['A', 'B', 'C1', 'C2', 'D', 'E', 'F']}
def snap(img, s):
    pal, lut = LUTS[s]
    a = (np.clip(img, 0, 255).astype(np.uint8) >> 2).astype(np.int32)
    return pal[lut[(a[..., 0] * 64 + a[..., 1]) * 64 + a[..., 2]]]

# ------------------------------------------------ effects
class Rain:
    def __init__(s, n, seed, spd, ln, alpha, col, slant=0.18):
        r = np.random.default_rng(seed)
        s.x = r.random(n) * (W + 60); s.y = r.random(n) * H
        s.sp = spd[0] + r.random(n) * (spd[1] - spd[0]); s.ln = (ln[0] + r.random(n) * (ln[1] - ln[0])).astype(int)
        s.a = alpha; s.col = C(*col); s.sl = slant
    def draw(s, img, f, mask=None, k=1.0):
        y = (s.y + s.sp * f) % (H + 16) - 8
        x = (s.x - s.sl * s.sp * f) % (W + 60) - 30
        for i in range(len(s.x)):
            for j in range(s.ln[i]):
                px = int(x[i] + s.sl * j); py = int(y[i] - j)
                if 0 <= px < W and 0 <= py < H and (mask is None or mask[py, px]):
                    a = s.a * k * (0.4 + 0.6 * j / s.ln[i])
                    img[py, px] = img[py, px] * (1 - a) + s.col * a

RAIN_F = Rain(150, 1, (6, 9), (3, 6), 0.35, (120, 135, 170))
RAIN_N = Rain(55, 2, (11, 15), (7, 12), 0.45, (160, 175, 205))

class Drips:
    """rain running down window glass: slow 1px trails with a bright head"""
    def __init__(s, n, seed, x0, x1, y0, y1):
        r = np.random.default_rng(seed)
        s.x = (x0 + r.random(n) * (x1 - x0)).astype(int); s.y = y0 + r.random(n) * (y1 - y0)
        s.sp = 0.15 + r.random(n) * 0.5; s.ln = 2 + r.integers(0, 6, n); s.y0, s.y1 = y0, y1
    def draw(s, img, f, mask=None):
        for i in range(len(s.x)):
            yy = int(s.y0 + (s.y[i] - s.y0 + s.sp[i] * f) % (s.y1 - s.y0))
            for j in range(s.ln[i]):
                py = yy - j; px = s.x[i]
                if 0 <= py < H and 0 <= px < W and (mask is None or mask[py, px]):
                    if j == 0: img[py, px] = img[py, px] * 0.3 + C(200, 210, 230) * 0.7
                    else: img[py, px] = img[py, px] * 0.65 + C(120, 135, 170) * 0.35

def glow(img, cx, cy, r, col, a):
    d2 = ((XX - cx) ** 2 + (YY - cy) ** 2) / (r * r)
    m = np.clip(1 - d2, 0, 1) ** 2 * a
    m = np.floor(m * 5) / 5            # banded glow (pixel-art style, no smooth gradient)
    img += C(*col) * m[..., None]

def flash(img, k, mask=None):
    lum = img.mean(2, keepdims=True) / 255
    add = (C(90, 105, 140) + C(120, 125, 140) * lum) * k
    if mask is not None: add = add * mask[..., None]
    img += add

def ribbon(img, ex, ey, t, f, length=45, phase=0.0):
    """1px cigarette smoke wisp"""
    for k in range(3, length):
        yk = ey - k
        if yk < 0: break
        xk = int(round(ex - k * 0.08 + math.sin(k * 0.17 - t * 3.0 + phase) * (0.4 + k * 0.08)))
        if (k + f // 3) % 15 in (0, 1) and k > 16: continue
        if 0 <= xk < W:
            c = C(190, 196, 214) if k < 12 else (C(140, 148, 170) if k < 28 else C(96, 104, 128))
            a = 0.85 if k < 28 else 0.6
            img[yk, xk] = img[yk, xk] * (1 - a) + c * a

def pulse_mask(img, mask, k):
    img[mask] = np.clip(img[mask] * k, 0, 255)

def dissolve(a, b, u):
    """ordered-dither dissolve between two keyframes (pixel-art friendly)"""
    h, w = a.shape[:2]
    bt = np.tile(BAY, (h // 4 + 1, w // 4 + 1))[:h, :w]
    return np.where((bt < u)[..., None], b, a)

def shake(t, t0, amp=2, dur=0.3):
    if t0 <= t < t0 + dur:
        k = 1 - (t - t0) / dur
        return int(round(math.sin(t * 90) * amp * k)), int(round(math.cos(t * 70) * amp * k))
    return 0, 0

def view(src, ox, oy):
    h, w = src.shape[:2]
    ox = int(max(0, min(w - W, ox))); oy = int(max(0, min(h - H, oy)))
    return src[oy:oy + H, ox:ox + W].copy()

def red_mask(img, box):
    x0, y0, x1, y1 = box
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    m = (R > 150) & (G < 90) & (B < 90)
    bm = np.zeros(m.shape, bool); bm[y0:y1, x0:x1] = True
    return m & bm

# ------------------------------------------------ shot A: building tilt-up  (0 - 7.0)
A = IMG['1']                                   # 384 x 576
A_EYES = red_mask(A, (160, 55, 230, 95))
A_WIN = np.zeros(A.shape[:2], bool)
def shot_A(t, f):
    oy = lerp(A.shape[0] - H, 0, ease(seg(t, 0.4, 6.6)))
    src = A.copy()
    pulse_mask(src, A_EYES, 0.75 + 0.45 * (0.5 + 0.5 * math.sin(t * 3.2)))
    img = view(src, 0, oy)
    # wet street shimmer: wobble reflection rows
    ry0 = int(A.shape[0] - 70 - oy)
    for y in range(max(0, ry0), H):
        s = int(round(math.sin(y * 0.9 + t * 7) * 1.2))
        if s: img[y] = np.roll(img[y], s, 0)
    # beacon blink on the spire
    by = int(11 - oy)
    if 0 <= by < H and (t % 1.3) < 0.45:
        glow(img, 192, by, 9, (230, 30, 25), 0.9)
        img[by - 1:by + 1, 191:193] = C(255, 120, 100)
    fl = 0.0
    for (tc, pk, d) in [(0.55, 1.0, 0.14), (0.78, 0.6, 0.1), (4.1, 0.5, 0.12)]:
        if tc <= t < tc + d: fl = max(fl, pk * (1 - (t - tc) / d))
    if fl: flash(img, fl * 0.9)
    RAIN_F.draw(img, f); RAIN_N.draw(img, f)
    img *= ease(seg(t, 0.0, 1.1))
    return snap(img, 'A')

# ------------------------------------------------ shot B: the lit window (7.0 - 9.6)
B = IMG['2']
B_EYES = red_mask(B, (100, 0, 300, 45))
def shot_B(t, f):
    u = t - 7.0
    oy = lerp(0, B.shape[0] - H, ease(seg(u, 0.1, 2.4)))
    src = B.copy()
    pulse_mask(src, B_EYES, 0.8 + 0.4 * (0.5 + 0.5 * math.sin(t * 3.2)))
    dx, dy = shake(t, 7.0, 3, 0.35)
    img = view(src, 0 - dx, oy - dy)
    # window light flicker (lamp inside)
    win = (YY + oy > 84) & (YY + oy < 154) & (XX > 58) & (XX < 327)
    k = 1.0 + 0.05 * math.sin(t * 13) * math.sin(t * 7.3)
    img[win] *= k
    ribbon(img, 178, int(110 - oy), t, f, 26)
    RAIN_F.draw(img, f); RAIN_N.draw(img, f, k=1.2)
    return snap(img, 'B')

# ------------------------------------------------ shot C1: office wide (9.6 - 12.4)
C1 = IMG['3-1']
C1_WIN = np.zeros(C1.shape[:2], bool); C1_WIN[22:140, 228:384] = True
C1_WIN &= (C1[..., 2] > C1[..., 0] * 0.9)   # only the blue-ish window pixels
DR1 = Drips(40, 5, 228, 384, 22, 140)
def shot_C1(t, f):
    u = t - 9.6
    oy = lerp(8, 36, ease(seg(u, 0.0, 2.8)))
    src = C1.copy()
    fl = 0.0
    for (tc, pk, d) in [(11.3, 0.9, 0.12), (11.5, 0.5, 0.18)]:
        if tc <= t < tc + d: fl = max(fl, pk * (1 - (t - tc) / d))
    if fl: flash(src, fl, C1_WIN.astype(np.float32) * 1.0 + 0.25)
    img = view(src, 0, oy)
    DR1.draw(img, f, view(C1_WIN[..., None].astype(np.float32), 0, oy)[..., 0] > 0)
    # lamp flicker + ashtray smoke
    glow(img, 347, 117 - oy, 18, (255, 170, 60), 0.15 + 0.05 * math.sin(t * 11))
    ribbon(img, 284, int(116 - oy), t, f, 40)
    img *= 0.2 + 0.8 * ease(seg(u, 0.0, 0.25))
    return snap(img, 'C1')

# ------------------------------------------------ shot C2: tiger painting (12.4 - 14.2)
C2 = IMG['3-2']
C2_EYE = np.zeros(C2.shape[:2], bool)
_R, _G, _B = C2[..., 0], C2[..., 1], C2[..., 2]
C2_EYE[88:110, 100:125] = True
C2_EYE &= (_R > 180) & (_G > 90) & (_B < 90)
def shot_C2(t, f):
    u = t - 12.4
    oy = lerp(38, 12, ease(seg(u, 0.0, 1.8)))
    src = C2.copy()
    g = 0.5 + 0.5 * math.sin(u * 5)
    pulse_mask(src, C2_EYE, 1.0 + 0.5 * g)
    img = view(src, 0, oy)
    fl = 0.0
    for (tc, pk, d) in [(12.4, 1.0, 0.12), (12.62, 0.7, 0.16)]:
        if tc <= t < tc + d: fl = max(fl, pk * (1 - (t - tc) / d))
    if fl:
        side = np.clip((XX - 250) / 134, 0, 1)
        flash(img, fl, side)
    glow(img, 192, 14 - oy, 60, (255, 180, 80), 0.10 + 0.04 * math.sin(t * 9))
    # eye glint pixels
    ys, xs = np.nonzero(C2_EYE)
    if len(xs) and g > 0.6:
        ex, ey = int(xs.mean()), int(ys.mean() - oy)
        if 0 <= ey < H: img[ey, ex] = C(255, 238, 170)
    img *= 0.25 + 0.75 * ease(seg(u, 0.0, 0.2))
    return snap(img, 'C2')

# ------------------------------------------------ shot D: boss puts the cigarette in his mouth (14.2 - 17.2)
DOY = 28
D_WIN = np.zeros(IMG['4-1'].shape[:2], bool); D_WIN[0:135, :] = True
_d = IMG['4-1']
_yyD, _xxD = np.mgrid[0:_d.shape[0], 0:_d.shape[1]]
D_WIN &= (_d[..., 2] > _d[..., 0] * 1.05) & ((_xxD < 60) | ((_xxD > 330) & (_yyD < 135)) | (_yyD < 30) | ((_xxD > 60) & (_xxD < 110) & (_yyD < 130)))
DR2 = Drips(34, 7, 0, 384, 0, 130)
KEYS_D = [(14.2, '4-1'), (15.25, '4-2'), (15.95, '4-3')]
def keyframe(t, keys, dis=0.13):
    cur = keys[0][1]
    for i, (tk, n) in enumerate(keys):
        if t >= tk: cur = n; idx = i
    tk, n = keys[idx]
    if idx > 0 and t < tk + dis:
        return dissolve(IMG[keys[idx - 1][1]], IMG[n], (t - tk) / dis)
    return IMG[n].copy()
def shot_D(t, f):
    src = keyframe(t, KEYS_D)
    fl = 0.0
    for (tc, pk, d) in [(16.55, 0.8, 0.12), (16.75, 0.45, 0.15)]:
        if tc <= t < tc + d: fl = max(fl, pk * (1 - (t - tc) / d))
    if fl: flash(src, fl, D_WIN.astype(np.float32) + 0.15)
    img = view(src, 0, DOY)
    DR2.draw(img, f, view(D_WIN[..., None].astype(np.float32), 0, DOY)[..., 0] > 0)
    # lamp flicker
    glow(img, 30, 100 - DOY, 34, (255, 160, 60), 0.10 + 0.05 * math.sin(t * 12.7))
    img *= 0.25 + 0.75 * ease(seg(t, 14.2, 14.45))
    return snap(img, 'D')

# ------------------------------------------------ shot E: lighter (17.2 - 19.4)
EOY = 20
def flame(img, bx, by, t, f, k=1.0):
    """pixel flame sprite, 4-step palette, flickering"""
    hgt = 26 + int(3 * math.sin(f * 1.9)) + (f % 3)
    lean = 0.18 + 0.06 * math.sin(f * 1.3)
    for j in range(hgt):
        u = j / hgt
        wdt = 5.5 * math.sin(math.pi * min(1, u * 1.15 + 0.05)) * (1 - u * 0.35)
        cx = bx + lean * j + math.sin(j * 0.6 + f * 0.9) * 0.4 * u
        for dx in range(-6, 7):
            d = abs(dx - (cx - bx - lean * j) ) if False else abs((bx + dx) - cx)
            if d <= wdt:
                r = d / max(wdt, 0.1)
                if u < 0.35 and r < 0.45: col = C(255, 255, 235)
                elif r < 0.6 and u < 0.75: col = C(255, 238, 170)
                elif r < 0.9: col = C(255, 170, 40)
                else: col = C(255, 90, 20)
                py = int(by - j); px = int(bx + dx)
                if 0 <= py < H and 0 <= px < W: img[py, px] = col
    # blue base
    if 0 <= by < H: img[by, int(bx) - 1:int(bx) + 2] = C(118, 126, 170)
def shot_E(t, f):
    if t < 17.8: src = IMG['5-1'].copy()
    elif t < 18.9: src = IMG['5-2'].copy()
    else: src = IMG['5-4'].copy()
    if 17.8 <= t < 17.9: src = dissolve(IMG['5-1'], IMG['5-2'], (t - 17.8) / 0.1)
    if 18.9 <= t < 19.0: src = dissolve(IMG['5-2'], IMG['5-4'], (t - 18.9) / 0.1)
    img = view(src, 0, EOY)
    if 18.2 <= t < 18.9:
        # sparks on the strike, then steady flame; warm light floods hand/lips
        bx, by = 176, 78 - EOY
        k = min(1.0, (t - 18.2) / 0.06)
        d2 = ((XX - bx) ** 2 + (YY - by + 8) ** 2) / (130 ** 2)
        warm = np.clip(1 - d2, 0, 1) ** 2.2 * k * (0.9 + 0.1 * math.sin(f * 2.1))
        lit = (img.mean(2) > 38)[..., None]
        img = np.where(lit, img * (1 + 1.4 * warm[..., None]) * (1 - warm[..., None] * C(0.0, 0.18, 0.42)), img)
        flame(img, bx, by, t, f)
        if t < 18.32:
            for _ in range(10):
                sx = int(bx + rng.normal() * 6); sy = int(by - 3 + rng.normal() * 5)
                if 0 <= sx < W and 0 <= sy < H: img[sy, sx] = C(255, 238, 170)
    if t >= 18.9:
        e = 0.5 + 0.5 * math.sin(t * 6)
        glow(img, 187, 72 - EOY, 10, (255, 110, 30), 0.35 + 0.3 * e)
        ribbon(img, 186, 70 - EOY, t, f, 34)
    img *= 0.25 + 0.75 * ease(seg(t, 17.2, 17.4))
    return snap(img, 'E')

# ------------------------------------------------ shot F: inhale - hold - exhale - stare (19.4 - 24.5)
FOY = 22
KEYS_F = [(19.4, '6-1'), (20.6, '6-2'), (21.1, '6-3'), (21.7, '6-4'), (22.6, '6-5')]
BGF = IMG['6-2']
def smoke_layer(n):
    a = IMG[n]; d = np.abs(a - BGF).mean(2)
    m = (d > 18) & (XX_F < 262) & (a.mean(2) > 55)
    return m
_hF = IMG['6-1'].shape[0]
XX_F = np.mgrid[0:_hF, 0:W][1]
SML = {n: smoke_layer(n) for n in ['6-3', '6-4', '6-5']}
F_WIN = (XX_F < 215) & (np.mgrid[0:_hF, 0:W][0] < 185) & (BGF[..., 2] > BGF[..., 0] * 1.05)
DR3 = Drips(40, 9, 0, 214, 0, 185)
EMB = {'6-1': (324, 159), '6-2': (332, 157), '6-3': (333, 157), '6-4': (333, 157), '6-5': (333, 157)}
def shot_F(t, f):
    cur = [n for (tk, n) in KEYS_F if t >= tk][-1]
    src = keyframe(t, KEYS_F, 0.12)
    # drifting smoke: re-composite the painted smoke layer with a slow drift over the smoke-free plate
    if cur in SML:
        tk = dict((n, tk) for tk, n in KEYS_F)[cur]
        dt = t - tk
        dxs = -int(dt * 7); dys = -int(dt * 2.5)
        m = SML[cur]
        base = src.copy()
        base[m] = BGF[m]
        sh = np.roll(np.roll(src, dys, 0), dxs, 1); mm = np.roll(np.roll(m, dys, 0), dxs, 1)
        base[mm] = sh[mm]
        src = base
    fl = 0.0
    for (tc, pk, d) in [(22.6, 1.0, 0.1), (22.8, 0.6, 0.18)]:
        if tc <= t < tc + d: fl = max(fl, pk * (1 - (t - tc) / d))
    if fl: flash(src, fl, F_WIN.astype(np.float32) + 0.2)
    img = view(src, 0, FOY)
    DR3.draw(img, f, view(F_WIN[..., None].astype(np.float32), 0, FOY)[..., 0] > 0)
    ex, ey = EMB[cur]; ey -= FOY
    if cur == '6-1':
        e = 0.6 + 0.4 * ease(seg(t, 19.5, 20.4)) + 0.1 * math.sin(f * 2.2)   # long drag: ember brightens
    else:
        e = 0.45 + 0.1 * math.sin(t * 5)
    glow(img, ex, ey, 16, (255, 110, 30), 0.45 * e)
    glow(img, ex - 8, ey - 6, 60, (120, 45, 0), 0.25 * e)                       # warm light on fingers/lips
    if cur != '6-1': ribbon(img, ex, ey - 2, t, f, 40)
    # fade out: everything to black, the ember last
    fo = 1 - ease(seg(t, 23.55, 24.25))
    img *= fo
    if fo < 1:
        ef = 1 - ease(seg(t, 24.1, 24.45))
        glow(img, ex, ey, 6, (255, 120, 30), 0.9 * ef)
        if ef > 0.1: img[int(ey), int(ex)] = C(255, 170, 40) * ef + img[int(ey), int(ex)] * (1 - ef)
    img *= 0.25 + 0.75 * ease(seg(t, 19.4, 19.55))
    return snap(img, 'F')

def render(t, f):
    if t < 7.0: return shot_A(t, f)
    if t < 9.6: return shot_B(t, f)
    if t < 12.4: return shot_C1(t, f)
    if t < 14.2: return shot_C2(t, f)
    if t < 17.2: return shot_D(t, f)
    if t < 19.4: return shot_E(t, f)
    return shot_F(t, f)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'prev':
        ts = [float(x) for x in sys.argv[2:]]
        sheet = Image.new('RGB', (W * 3, H * ((len(ts) + 2) // 3)))
        for i, tt in enumerate(ts):
            sheet.paste(Image.fromarray(render(tt, int(tt * FPS))), ((i % 3) * W, (i // 3) * H))
        sheet.resize((sheet.width * 2, sheet.height * 2), Image.NEAREST).save('prev.png')
        sys.exit()
    S = 5
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    p = subprocess.Popen([ff, '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W*S}x{H*S}', '-r', str(FPS), '-i', '-', '-i', 'intro3.wav',
                          '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-tune', 'animation', '-pix_fmt', 'yuv420p',
                          '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', 'paedo_intro_v3.mp4'], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for f in range(NF):
        fr = render(f / FPS, f)
        p.stdin.write(fr.repeat(S, 0).repeat(S, 1).tobytes())
        if f % 48 == 0: print('frame', f, flush=True)
    p.stdin.close(); p.wait(); print('done', p.returncode)
