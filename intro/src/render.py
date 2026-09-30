"""PAEDO intro - pixel art renderer (480x270 native, 24fps)."""
import numpy as np
from PIL import Image
import math, sys

W, H = 480, 270
FPS = 24
DUR = 14.0
NF = int(DUR * FPS)
rng = np.random.default_rng(7)

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)

# ---------------------------------------------------------------- helpers
def C(*v):
    return np.array(v, np.float32)

def clamp01(x):
    return min(1.0, max(0.0, x))

def ease_io(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)

def ease_in(x):
    x = clamp01(x)
    return x * x * x

def lerp(a, b, u):
    return a + (b - a) * u

def seg(t, a, b):
    return clamp01((t - a) / (b - a))

def rect(img, x0, y0, x1, y1, col, a=1.0):
    x0 = int(max(0, math.floor(x0))); y0 = int(max(0, math.floor(y0)))
    x1 = int(min(W, math.floor(x1))); y1 = int(min(H, math.floor(y1)))
    if x1 <= x0 or y1 <= y0:
        return
    if a >= 1:
        img[y0:y1, x0:x1] = col
    else:
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + col * a

def blend(img, mask, col, a=1.0):
    """mask float HxW 0..1"""
    m = (mask * a)[..., None]
    img[:] = img * (1 - m) + col * m

def ellipse_mask(cx, cy, rx, ry):
    return (((XX + 0.5 - cx) / rx) ** 2 + ((YY + 0.5 - cy) / ry) ** 2) <= 1.0

def poly_mask(pts):
    pts = np.asarray(pts, np.float32)
    x = XX + 0.5; y = YY + 0.5
    inside = np.zeros((H, W), bool)
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]; xj, yj = pts[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
        inside ^= cond
        j = i
    return inside

BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) + 0.5) / 16
BAYER_T = np.tile(BAYER, (H // 4 + 1, W // 4 + 1))[:H, :W]

def dith(alpha):
    """hard pixel-art alpha via ordered dithering"""
    return (alpha > BAYER_T).astype(np.float32)

def glow(img, cx, cy, r, col, a):
    d2 = ((XX - cx) ** 2 + (YY - cy) ** 2) / (r * r)
    m = np.exp(-d2 * 2.5) * a
    img[:] = img + col * m[..., None]

def edge_rim(mask, dx, dy):
    sh = np.zeros_like(mask)
    ys = slice(max(0, dy), H + min(0, dy)); yd = slice(max(0, -dy), H + min(0, -dy))
    xs = slice(max(0, dx), W + min(0, dx)); xd = slice(max(0, -dx), W + min(0, -dx))
    sh[yd, xd] = mask[ys, xs]
    return mask & ~sh

def value_noise(w, h, scales, seed):
    r = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp = 1.0; tot = 0
    for s in scales:
        g = r.random((max(2, h // s + 2), max(2, w // s + 2))).astype(np.float32)
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((w + 2 * s, h + 2 * s), Image.BICUBIC)
        a = np.asarray(im, np.float32)[s:s + h, s:s + w] / 255
        out += a * amp; tot += amp; amp *= 0.5
    return out / tot

CLOUD = value_noise(1400, 700, [96, 48, 24, 12], 3)

# ---------------------------------------------------------------- lightning
def flash_at(t):
    f = 0.0
    for (tc, pk, dur) in [(0.45, 0.55, 0.28), (7.30, 0.35, 0.08), (7.44, 1.0, 0.5), (12.62, 0.8, 0.16), (12.84, 0.5, 0.22)]:
        if tc <= t < tc + dur:
            u = (t - tc) / dur
            f = max(f, pk * (1 - u) ** 2)
    return f

# ---------------------------------------------------------------- rain
class Rain:
    def __init__(self, n, seed, spd, ln, alpha, col):
        r = np.random.default_rng(seed)
        self.x = r.random(n) * (W + 80); self.y = r.random(n) * H
        self.sp = spd[0] + r.random(n) * (spd[1] - spd[0])
        self.ln = ln[0] + r.random(n) * (ln[1] - ln[0])
        self.alpha = alpha; self.col = col
    def draw(self, img, frame, k=1.0):
        y = (self.y + self.sp * frame) % (H + 20) - 10
        x = (self.x - 0.28 * self.sp * frame) % (W + 80) - 40
        for i in range(len(self.x)):
            L = int(self.ln[i])
            for j in range(L):
                px = int(x[i] + 0.28 * j); py = int(y[i] - j)
                if 0 <= px < W and 0 <= py < H:
                    a = self.alpha * k * (j / L)
                    img[py, px] = img[py, px] * (1 - a) + self.col * a

RAIN_FAR = Rain(170, 11, (7, 10), (3, 6), 0.30, C(0.42, 0.47, 0.58))
RAIN_NEAR = Rain(70, 12, (13, 18), (8, 14), 0.45, C(0.62, 0.68, 0.78))

# ---------------------------------------------------------------- exterior
from tiger import tiger as make_tiger

def gen_layer(seed, xr, wr, hr, keep=None, rowp=0.3, winp=0.55):
    r = np.random.default_rng(seed)
    bl = []
    x = xr[0]
    while x < xr[1]:
        w = int(r.integers(wr[0], wr[1])); h = int(r.integers(hr[0], hr[1]))
        rows = h // 4 + 1; cols = w // 3 + 1
        rowr = r.random(rows)
        rr = r.random((rows, cols))
        lit = ((rowr[:, None] < rowp) & (rr < winp)) | (rr < 0.02)
        tint = r.random((rows, cols))
        if keep is None or keep(x, w):
            bl.append((x, w, h, lit, tint, int(r.integers(0, 3))))
        x += w + int(r.integers(-6, 5))
    return bl

FAR = gen_layer(21, (-900, 900), (22, 55), (40, 175), rowp=0.25, winp=0.35)
MID = gen_layer(22, (-800, 800), (30, 70), (40, 235), rowp=0.25, winp=0.45)
NEAR = gen_layer(23, (-700, 700), (50, 110), (140, 340), keep=lambda x, w: x + w < -85 or x > 85, rowp=0.3, winp=0.5)

TW = np.random.default_rng(31).random((100, 20))

class Cam:
    def __init__(self, cx, cy, z):
        self.cx, self.cy, self.z = cx, cy, z
    def layer(self, p):
        return Cam(self.cx * p, self.cy * p, 1 + (self.z - 1) * p)
    def sx(self, x):
        return W / 2 + (x - self.cx) * self.z
    def sy(self, y):
        return H / 2 - (y - self.cy) * self.z

def wrect(img, cam, x0, y0, x1, y1, col, a=1.0):
    sx0 = math.floor(cam.sx(x0)); sx1 = math.floor(cam.sx(x1))
    sy0 = math.floor(cam.sy(y1)); sy1 = math.floor(cam.sy(y0))
    if sx1 == sx0: sx1 = sx0 + 1
    if sy1 == sy0: sy1 = sy0 + 1
    rect(img, sx0, sy0, sx1, sy1, col, a)

def draw_layer(img, cam, bl, base_col, win_col, fl, wsz=(1, 2)):
    for (x, w, h, lit, tint, kind) in bl:
        sx0 = cam.sx(x); sx1 = cam.sx(x + w)
        if sx1 < 0 or sx0 > W or cam.sy(h) > H:
            continue
        col = base_col * (1 + 0.18 * kind) + fl * C(0.10, 0.11, 0.16)
        wrect(img, cam, x, -400, x + w, h, col)
        wrect(img, cam, x, h - 1, x + w, h, col * 1.6)
        if kind == 2:
            wrect(img, cam, x + w / 2, h, x + w / 2 + 1, h + 14, col)
            if cam.z < 2:
                wrect(img, cam, x + w / 2, h + 14, x + w / 2 + 1, h + 15, C(0.7, 0.08, 0.06))
        rows, cols = lit.shape
        for ri in range(rows):
            wy = h - 4 - ri * 4
            if wy < -30: break
            syy = cam.sy(wy)
            if syy < -4 or syy > H + 4: continue
            for ci in range(cols - 1):
                if lit[ri, ci]:
                    wx = x + 1 + ci * 3
                    if wx + wsz[0] > x + w - 1: break
                    tt = tint[ri, ci]
                    wc = win_col * (0.7 + 0.5 * tt) if tt > 0.3 else C(0.45, 0.55, 0.65) * (0.8 + tt)
                    wrect(img, cam, wx, wy, wx + wsz[0], wy + wsz[1] - 1 + 0.9, wc)

TIGER_ICON = [
    "X..X.....X..X",
    "XXXX.....XXXX",
    ".XXXXXXXXXXX.",
    "XX.X.XXX.X.XX",
    "XXXXXXXXXXXXX",
    "XRRXXXXXXXRRX",
    "XXXXX...XXXXX",
    ".XXXX.X.XXXX.",
    "..XXX...XXX..",
    "...XXXXXXX...",
    ".....XXX.....",
]
LED = C(0.55, 0.72, 0.85)
OFFICE = (6, 30, 567, 578)

def draw_tower(img, cam, t, fl):
    body = C(0.045, 0.05, 0.068) + fl * C(0.10, 0.11, 0.15)
    side = C(0.026, 0.028, 0.04) + fl * C(0.05, 0.05, 0.08)
    wrect(img, cam, -50, -2, 50, 560, body)
    wrect(img, cam, 50, -2, 64, 556, side)
    wrect(img, cam, -42, 560, 42, 596, body)
    wrect(img, cam, 42, 560, 52, 593, side)
    wrect(img, cam, -30, 596, 30, 628, body * 1.1)
    wrect(img, cam, 30, 596, 37, 625, side)
    wrect(img, cam, -0.5, 628, 1.5, 690, side * 1.8)
    # glass floors
    sheen_y = 150 + (t * 55) % 480
    for k in range(78):
        y0 = 7 * k + 12
        if cam.sy(y0) < -12 or cam.sy(y0 + 5) > H + 12:
            continue
        for i in range(16):
            x0 = -48 + i * 6
            r = TW[k, i]
            if k < 22:
                lit = r < 0.66; col = C(0.66, 0.72, 0.78) if TW[k, 19] < 0.5 else C(0.80, 0.70, 0.48)
            elif k < 58:
                lit = (TW[k, 18] < 0.25 and r < 0.5) or r < 0.03; col = C(0.58, 0.64, 0.72)
            else:
                lit = False; col = None
            if lit:
                col = col * (0.8 + 0.3 * TW[k, 17 - i])
            else:
                g = 0.06 + 0.02 * (k / 78) + 0.015 * r
                d = abs(y0 - sheen_y + x0 * 1.2)
                g += 0.045 * max(0, 1 - d / 26)
                col = C(g * 0.8, g * 0.9, g * 1.3) + fl * C(0.25, 0.28, 0.38) * (0.6 + r)
            wrect(img, cam, x0, y0, x0 + 5, y0 + 5, col)
        for i in range(3):
            wrect(img, cam, 51.5 + i * 4, y0, 54 + i * 4, y0 + 5, C(0.038, 0.042, 0.06) + fl * 0.12)
    # lobby
    wrect(img, cam, -48, 0, 48, 10, C(0.85, 0.74, 0.52))
    for i in range(9):
        wrect(img, cam, -48 + i * 12, 0, -47 + i * 12, 10, C(0.10, 0.08, 0.06))
    wrect(img, cam, -8, 0, 8, 7, C(1.0, 0.9, 0.7))
    # office band
    wrect(img, cam, -40, 565, 40, 580, C(0.05, 0.055, 0.078) + fl * C(0.2, 0.22, 0.3))
    for x0 in range(-40, 40, 8):
        wrect(img, cam, x0, 565, x0 + 0.6, 580, C(0.02, 0.02, 0.03))
    wrect(img, cam, -40, 584, 40, 591, C(0.055, 0.06, 0.08) + fl * 0.2)
    draw_office_window(img, cam, t)
    # crown fins + emblem
    for x0 in range(-28, 30, 6):
        wrect(img, cam, x0, 598, x0 + 1, 626, C(0.08, 0.085, 0.11) + fl * 0.2)
    for r_, row in enumerate(TIGER_ICON):
        for c_, ch in enumerate(row):
            if ch == '.': continue
            wx = -13 + c_ * 2; wy = 623 - r_ * 2
            if ch == 'R':
                col = C(0.85, 0.08, 0.05) * (0.8 + 0.2 * math.sin(t * 4))
            else:
                col = C(0.17, 0.06, 0.055) + fl * 0.25
            wrect(img, cam, wx, wy - 2, wx + 2, wy, col)
    # LED edge lines (architectural lighting)
    led = LED * (0.85 + fl * 0.3)
    wrect(img, cam, -50, 0, -49, 560, led)
    wrect(img, cam, 49, 0, 50, 560, led * 0.9)
    wrect(img, cam, 63, 0, 64, 556, led * 0.35)
    wrect(img, cam, -42, 560, -41, 596, led); wrect(img, cam, 41, 560, 42, 596, led * 0.9)
    wrect(img, cam, -42, 595, 42, 596, led)
    wrect(img, cam, -50, 559, 50, 560, led)
    wrect(img, cam, -30, 596, -29, 628, led); wrect(img, cam, 29, 596, 30, 628, led * 0.9)
    wrect(img, cam, -30, 627, 30, 628, led)
    glow_w = 30 * cam.z
    # beacon
    if (t % 1.2) < 0.4:
        sx, sy = cam.sx(0.5), cam.sy(690)
        glow(img, sx, sy, 12 * cam.z, C(0.9, 0.05, 0.05), 0.7)
        wrect(img, cam, -0.5, 689, 1.5, 691, C(1, 0.25, 0.2))

def draw_office_window(img, cam, t):
    ox0, ox1, oy0, oy1 = OFFICE
    sx0, sx1 = cam.sx(ox0), cam.sx(ox1); sy0, sy1 = cam.sy(oy1), cam.sy(oy0)
    x0i, x1i = int(max(0, math.floor(sx0))), int(min(W, math.floor(sx1)))
    y0i, y1i = int(max(0, math.floor(sy0))), int(min(H, math.floor(sy1)))
    if x1i <= x0i or y1i <= y0i:
        return
    sub = (slice(y0i, y1i), slice(x0i, x1i))
    wx = (XX[sub] + 0.5 - W / 2) / cam.z + cam.cx
    wy = -(YY[sub] + 0.5 - H / 2) / cam.z + cam.cy
    lamp = np.exp(-(((wx - 12) / 7) ** 2 + ((wy - 570) / 5) ** 2))
    base = 0.18 + 0.8 * lamp
    if cam.z > 3:
        base = np.floor(base * 6) / 6  # banded light
    col = C(1.0, 0.60, 0.24) * base[..., None]
    if cam.z > 2.2:
        dark = C(0.03, 0.018, 0.012)
        sil = np.zeros(wx.shape, bool)
        sil |= (wy < 569.6) & (wx > 9) & (wx < 28)                                   # desk
        chair = (((wx - 21) / 2.4) ** 2 + ((wy - 571.4) / 2.6) ** 2 <= 1) & (wy > 569)
        sh = (wy > 569) & (wy < 572.9) & (np.abs(wx - 21) < 3.0 - np.clip(wy - 571.6, 0, 3) * 1.3)
        sil |= sh                                                                     # shoulders
        sil |= ((wx - 21) / 1.0) ** 2 + ((wy - 574.5) / 1.2) ** 2 <= 1             # head
        sil |= (np.abs(wx - 21) < 0.5) & (wy > 572) & (wy < 573.6)                     # neck
        # lamp
        lampm = (wy > 569.6) & (wy < 571.2) & (np.abs(wx - 12) < 0.9 + (571.2 - wy) * 0.5)
        pole = (np.abs(wx - 12) < 0.12) & (wy > 569.5) & (wy < 570.2)
        col = np.where(chair[..., None], C(0.09, 0.05, 0.03), col)
        col = np.where(sil[..., None], dark, col)
        emb = ((wx - 22.6) ** 2 + (wy - 573.9) ** 2) < 0.06
        col = np.where(emb[..., None], C(1.0, 0.35, 0.08), col)
        col = np.where(lampm[..., None], C(1.0, 0.72, 0.38), col)
        col = np.where(pole[..., None], dark, col)
        # smoke wisp
        sm = (np.abs(wx - 22.6 - np.sin(wy * 3 + t * 4) * 0.25) < 0.12) & (wy > 574.1) & (wy < 577)
        col = np.where(sm[..., None], col * 0.6 + C(0.5, 0.45, 0.42) * 0.4, col)
        # mullions
        for mx in (12.0 + 3.8, 22.0 + 3.8):
            m = np.abs(wx - mx) < 0.18
            col = np.where(m[..., None], C(0.02, 0.02, 0.025), col)
        # rain streaks on glass (world-anchored)
        rs = ((np.floor(wx * 3) * 7.3) % 1 < 0.035) & (((wy * 2 + t * 6 + np.floor(wx * 3) * 1.7) % 3) < 0.9)
        col = np.where(rs[..., None], col * 0.75 + C(0.6, 0.62, 0.7) * 0.25, col)
    img[sub] = col
    wrect(img, cam, ox0 - 0.3, oy0 - 0.3, ox1 + 0.3, oy0, C(0.03, 0.03, 0.04))
    wrect(img, cam, ox0 - 0.3, oy1, ox1 + 0.3, oy1 + 0.3, C(0.03, 0.03, 0.04))

def draw_street(img, cam, t, frame):
    if cam.sy(0) > H: return
    wrect(img, cam, -2000, -140, 2000, 0, C(0.035, 0.035, 0.045))
    wrect(img, cam, -2000, -3, 2000, 0, C(0.13, 0.13, 0.15))
    wrect(img, cam, -2000, -48, 2000, -45, C(0.11, 0.11, 0.13))
    # wet reflections of lobby / tower LEDs
    for d in range(0, 42):
        a = 0.35 * (1 - d / 42) ** 1.5
        wig = int(math.sin(d * 1.3 + t * 5) * 1.5)
        wrect(img, cam, -48 + wig, -4 - d, 48 + wig, -3 - d, C(0.8, 0.68, 0.45), a * (0.6 if d % 3 == 0 else 1))
        wrect(img, cam, -50 + wig, -4 - d, -49 + wig, -3 - d, LED, a)
        wrect(img, cam, 49 + wig, -4 - d, 50 + wig, -3 - d, LED, a)
    for x in range(-600, 600, 24):
        wrect(img, cam, x, -24, x + 10, -23, C(0.22, 0.21, 0.18))
    for (lane, spd, off, dirn) in [(-15, 70, 0, 1), (-35, 95, 180, -1), (-15, 60, 440, 1), (-35, 80, 560, -1)]:
        cxw = ((off + spd * t * dirn) % 900) - 450
        body = C(0.028, 0.028, 0.034)
        wrect(img, cam, cxw, lane, cxw + 22, lane + 6, body)
        wrect(img, cam, cxw + 5, lane + 6, cxw + 17, lane + 9, body)
        wrect(img, cam, cxw + 6, lane + 6.5, cxw + 16, lane + 8.5, C(0.09, 0.10, 0.14))
        wrect(img, cam, cxw, lane + 5, cxw + 22, lane + 6, C(0.12, 0.12, 0.15))
        hx, tx = (cxw + 22, cxw) if dirn > 0 else (cxw, cxw + 22)
        wrect(img, cam, hx - 1, lane + 2, hx, lane + 4, C(1, 0.95, 0.8))
        glow(img, cam.sx(hx + 8 * dirn), cam.sy(lane + 3), 16 * cam.z, C(0.9, 0.85, 0.6), 0.3)
        wrect(img, cam, tx - (1 if dirn < 0 else 0), lane + 2, tx + (0 if dirn < 0 else 1), lane + 4, C(0.95, 0.1, 0.06))
        glow(img, cam.sx(tx), cam.sy(lane + 3), 6 * cam.z, C(0.8, 0.05, 0.03), 0.3)
        for d in range(1, 7):
            wrect(img, cam, cxw + 3, lane - d, cxw + 19, lane - d + 1, C(0.5, 0.45, 0.35), 0.16 * (1 - d / 7))
    for x in (-160, -100, 100, 160):
        wrect(img, cam, x, 0, x + 1, 26, C(0.07, 0.07, 0.08))
        wrect(img, cam, x - 4, 25, x + 1, 26.5, C(0.07, 0.07, 0.08))
        wrect(img, cam, x - 4, 24, x - 1, 25, C(1, 0.72, 0.35))
        glow(img, cam.sx(x - 2.5), cam.sy(24), 12 * cam.z, C(1, 0.6, 0.2), 0.4)
        glow(img, cam.sx(x - 2.5), cam.sy(-2), 14 * cam.z, C(0.6, 0.35, 0.12), 0.3)

def draw_sky(img, cam, t, fl):
    off = cam.cy * 0.10
    v = np.clip((YY + off - 10) / 270, 0, 1)
    top = C(0.012, 0.014, 0.032); hor = C(0.12, 0.075, 0.105)
    g = v ** 1.5
    g = np.floor(g * 10 + BAYER_T * 0.999) / 10   # banded + ordered dither on band edges only
    img[:] = top + (hor - top) * g[..., None]
    ox = int(t * 8) % 700; oy = int(np.clip(300 - cam.cy * 0.4, 0, 420))
    cl = CLOUD[oy:oy + H, ox:ox + W]
    lvl1 = (cl > 0.50).astype(np.float32)
    lvl2 = (cl > 0.58).astype(np.float32)
    lvl3 = (cl > 0.66).astype(np.float32)
    blend(img, lvl1, C(0.055, 0.052, 0.085) + fl * C(0.30, 0.32, 0.44), 1)
    blend(img, lvl2, C(0.08, 0.072, 0.11) + fl * C(0.45, 0.48, 0.62), 1)
    blend(img, lvl3, C(0.11, 0.095, 0.135) + fl * C(0.6, 0.62, 0.78), 1)
    # moon behind clouds (upper right)
    my = 55 + (600 - cam.cy) * 0.08
    moon = ellipse_mask(395, my, 13, 13).astype(np.float32) * (1 - lvl2 * 0.85)
    blend(img, moon, C(0.62, 0.62, 0.66), 1)
    blend(img, (moon * ellipse_mask(400, my - 3, 11, 11)), C(0.48, 0.48, 0.54), 1)
    glow(img, 395, my, 70, C(0.10, 0.10, 0.14), 0.6)
    img += fl * C(0.08, 0.09, 0.13)

BOLT = None
def draw_bolt(img, t):
    global BOLT
    if not (0.45 <= t < 0.62): return
    if BOLT is None:
        r = np.random.default_rng(5)
        pts = [(110.0, -5.0)]; x, y = 110.0, -5.0
        while y < 110:
            y += int(r.integers(4, 10)); x += int(r.integers(-7, 8)); pts.append((x, y))
        BOLT = pts
    for (a, b) in zip(BOLT[:-1], BOLT[1:]):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        for i in range(n):
            px = int(a[0] + (b[0] - a[0]) * i / n); py = int(a[1] + (b[1] - a[1]) * i / n)
            if 0 <= px < W and 0 <= py < H:
                img[py, px] = C(0.95, 0.95, 1.0)
                if px + 1 < W: img[py, px + 1] = C(0.6, 0.6, 0.8)
    glow(img, 110, 50, 90, C(0.3, 0.3, 0.45), 0.6)

def ext_cam(t):
    if t < 1.3:
        return Cam(-4, lerp(40, 70, ease_io(t / 1.3)), 1.0)
    if t < 5.0:
        u = ease_io(seg(t, 1.3, 5.0)); return Cam(lerp(-4, 4, u), lerp(70, 600, u), 1.0)
    u = seg(t, 5.0, 7.45)
    z = math.exp(lerp(0, math.log(14), ease_in(u)))
    return Cam(lerp(4, 18, ease_io(u)), lerp(600, 572.5, ease_io(u)), z)

def exterior(t, frame):
    img = np.zeros((H, W, 3), np.float32)
    cam = ext_cam(t)
    fl = flash_at(t) * (0.8 if t < 7 else 0.3)
    draw_sky(img, cam, t, fl)
    draw_bolt(img, t)
    draw_layer(img, cam.layer(0.25), FAR, C(0.045, 0.043, 0.068), C(0.45, 0.40, 0.32), fl * 0.6, wsz=(1, 1))
    draw_layer(img, cam.layer(0.5), MID, C(0.030, 0.031, 0.048), C(0.55, 0.50, 0.38), fl * 0.5)
    draw_layer(img, cam.layer(0.8), NEAR, C(0.018, 0.019, 0.028), C(0.62, 0.55, 0.40), fl * 0.4, wsz=(2, 2))
    draw_street(img, cam, t, frame)
    draw_tower(img, cam, t, fl)
    RAIN_FAR.draw(img, frame)
    RAIN_NEAR.draw(img, frame)
    img *= ease_io(seg(t, 0.0, 1.2))
    return img

# ---------------------------------------------------------------- interior
IW = 760
PAINT_W, PAINT_H = 120, 92
PAINT = make_tiger(PAINT_W, PAINT_H)
_r = np.random.default_rng(55)
CITY_LIGHTS = [(_r.random() * IW, 50 + _r.random() * 140, int(_r.integers(0, 4)), 1.5 + _r.random() * 2.5) for _ in range(45)]
CITY_BLD = gen_layer(66, (0, IW), (18, 50), (40, 160), rowp=0.3, winp=0.4)
DRIPS = [(_r.random() * IW, _r.random() * 180, 3 + int(_r.integers(0, 8)), 0.3 + _r.random() * 1.0) for _ in range(70)]

def boss_small(img, bx, by, light, rim, S=1.0):
    skin_l = C(0.80, 0.60, 0.50) * light
    skin_d = C(0.36, 0.25, 0.22) * (0.6 + 0.4 * light)
    black = C(0.028, 0.03, 0.04)
    chair = ellipse_mask(bx, by + 14 * S, 18 * S, 22 * S) | ((np.abs(XX + 0.5 - bx) < 18 * S) & (YY >= by + 14 * S) & (YY < by + 70 * S))
    blend(img, chair.astype(np.float32), C(0.05, 0.03, 0.025))
    blend(img, edge_rim(chair, 0, -1).astype(np.float32), C(0.075, 0.05, 0.04))
    torso = poly_mask([(bx - 25 * S, by + 70 * S), (bx - 24 * S, by + 34 * S), (bx - 16 * S, by + 26 * S), (bx - 6 * S, by + 22 * S), (bx + 6 * S, by + 22 * S), (bx + 16 * S, by + 26 * S), (bx + 24 * S, by + 34 * S), (bx + 25 * S, by + 70 * S)])
    blend(img, torso.astype(np.float32), black)
    for (a0, b0) in [((bx - 12 * S, by + 30 * S), (bx - 8 * S, by + 60 * S)), ((bx + 13 * S, by + 31 * S), (bx + 10 * S, by + 58 * S))]:
        n = int(30 * S)
        for i in range(n):
            px = a0[0] + (b0[0] - a0[0]) * i / n; py = a0[1] + (b0[1] - a0[1]) * i / n
            rect(img, px, py, px + 1, py + 1, C(0.065, 0.068, 0.088))
    neck = (np.abs(XX + 0.5 - bx) < 3.6 * S) & (YY >= by + 13 * S) & (YY < by + 25 * S)
    blend(img, neck.astype(np.float32), skin_d)
    v = poly_mask([(bx - 4.5 * S, by + 22 * S), (bx + 4.5 * S, by + 22 * S), (bx, by + 35 * S)])
    blend(img, v.astype(np.float32), skin_d * 1.1)
    blend(img, (v & (XX < bx)).astype(np.float32), skin_l * 0.75)
    cl = poly_mask([(bx - 7 * S, by + 21 * S), (bx - 4.5 * S, by + 22 * S), (bx, by + 35 * S), (bx - 4 * S, by + 33 * S)]) | poly_mask([(bx + 7 * S, by + 21 * S), (bx + 4.5 * S, by + 22 * S), (bx, by + 35 * S), (bx + 4 * S, by + 33 * S)])
    blend(img, cl.astype(np.float32), C(0.07, 0.075, 0.095))
    head = ellipse_mask(bx, by + 8.5 * S, 6.4 * S, 8.0 * S) | poly_mask([(bx - 6.2 * S, by + 9 * S), (bx - 4.6 * S, by + 15.5 * S), (bx - 1.4 * S, by + 18 * S), (bx + 1.4 * S, by + 18 * S), (bx + 4.6 * S, by + 15.5 * S), (bx + 6.2 * S, by + 9 * S)])
    blend(img, head.astype(np.float32), skin_d)
    blend(img, (head & (XX + 0.5 < bx + 0.8 * S)).astype(np.float32), skin_l)
    ears = ellipse_mask(bx - 6.6 * S, by + 9.5 * S, 1.2 * S, 2.0 * S) | ellipse_mask(bx + 6.6 * S, by + 9.5 * S, 1.2 * S, 2.0 * S)
    blend(img, ears.astype(np.float32), skin_d)
    hair = (ellipse_mask(bx + 0.6 * S, by + 3.0 * S, 7.0 * S, 4.6 * S) & (YY < by + 3.6 * S + (XX - bx) * 0.12))
    hair |= ellipse_mask(bx - 6.0 * S, by + 5.0 * S, 1.2 * S, 2.4 * S) | ellipse_mask(bx + 6.2 * S, by + 5.0 * S, 1.2 * S, 2.4 * S)
    hair |= (np.abs(XX + 0.5 - (bx - 2.2 * S) - (YY - by - 3.5 * S) * 0.35) < 0.55) & (YY > by + 3 * S) & (YY < by + 6.6 * S)
    hair |= (np.abs(XX + 0.5 - (bx + 1.2 * S) - (YY - by - 3.5 * S) * 0.25) < 0.5) & (YY > by + 3 * S) & (YY < by + 5.6 * S)
    blend(img, hair.astype(np.float32), C(0.018, 0.02, 0.03))
    hl = edge_rim(hair, 0, -1) & (XX < bx + 2 * S)
    blend(img, hl.astype(np.float32), C(0.10, 0.10, 0.14) + rim * 0.3)
    rect(img, bx - 5 * S, by + 7 * S, bx - 1.5 * S, by + 8 * S, C(0.03, 0.02, 0.02))
    rect(img, bx + 1.5 * S, by + 7 * S, bx + 5 * S, by + 8 * S, C(0.02, 0.015, 0.015))
    rect(img, bx - 4 * S, by + 9 * S, bx - 2 * S, by + 10 * S, C(0.05, 0.03, 0.03))
    rect(img, bx + 2 * S, by + 9 * S, bx + 4 * S, by + 10 * S, C(0.02, 0.015, 0.015))
    rect(img, bx + 0.5 * S, by + 10 * S, bx + 1.5 * S, by + 13.5 * S, skin_d * 0.7)
    rect(img, bx - 2 * S, by + 15 * S, bx + 2 * S, by + 16 * S, skin_d * 0.6)
    # arm + hand at chin
    arm = poly_mask([(bx - 22 * S, by + 70 * S), (bx - 15 * S, by + 70 * S), (bx - 5 * S, by + 21 * S), (bx - 11 * S, by + 20 * S)])
    blend(img, arm.astype(np.float32), C(0.05, 0.052, 0.07))
    blend(img, edge_rim(arm, 1, 0).astype(np.float32), C(0.09, 0.095, 0.12))
    hand = ellipse_mask(bx - 3.2 * S, by + 16.8 * S, 3.8 * S, 3.0 * S)
    blend(img, hand.astype(np.float32), skin_l * 0.9)
    blend(img, (hand & (YY > by + 17.5 * S)).astype(np.float32), skin_d)
    rect(img, bx - 10.5 * S, by + 20 * S, bx - 5 * S, by + 22.5 * S, C(0.55, 0.57, 0.62))
    rect(img, bx - 9 * S, by + 20.5 * S, bx - 7 * S, by + 22 * S, C(0.05, 0.05, 0.06))
    rect(img, bx - 0.5 * S, by + 15 * S, bx + 6 * S, by + 16 * S, C(0.88, 0.86, 0.8))
    # cool window rim on right edges
    body = head | hair | torso | chair
    blend(img, edge_rim(head | hair | torso, 1, 0).astype(np.float32), C(0.20, 0.23, 0.32) + rim * C(0.5, 0.5, 0.6), 0.85)

def interior(t, frame):
    img = np.zeros((H, W, 3), np.float32)
    camx = lerp(0, IW - W, ease_io(seg(t, 7.45, 10.45)))
    fl = flash_at(t) * 0.6
    wx0, wx1 = 330 - camx, IW - camx
    pc = camx * 0.55
    view = np.zeros((H, W, 3), np.float32)
    v = np.clip((YY - 20) / 185, 0, 1)
    g = np.floor((v ** 1.4) * 8 + BAYER_T * 0.999) / 8
    view[:] = C(0.018, 0.02, 0.038) + (C(0.10, 0.065, 0.10) - C(0.018, 0.02, 0.038)) * g[..., None]
    view += fl * C(0.3, 0.32, 0.42)
    for (x, w, h, lit, tint, kind) in CITY_BLD:
        sx = x + 200 - pc
        if sx > W or sx + w < 0: continue
        rect(view, sx, 205 - h, sx + w, 205, C(0.028, 0.028, 0.042) + fl * 0.15)
        rows, cols = lit.shape
        for ri in range(rows):
            for ci in range(cols - 1):
                if lit[ri, ci]:
                    rect(view, sx + 1 + ci * 3, 205 - h + 2 + ri * 4, sx + 2 + ci * 3, 205 - h + 3 + ri * 4, C(0.6, 0.5, 0.35) * (0.6 + 0.6 * tint[ri, ci]))
    for (x, y, k, r) in CITY_LIGHTS:
        sx = (x + 100 - pc) % IW
        col = [C(0.9, 0.6, 0.25), C(0.45, 0.55, 0.85), C(0.85, 0.15, 0.1), C(0.8, 0.8, 0.7)][k]
        m = ellipse_mask(sx, y, r, r).astype(np.float32)
        blend(view, m, col, 0.22)
    for (x, y, L, sp) in DRIPS:
        sx = x - camx; yy = (y + sp * frame) % 190 + 15
        rect(view, sx, yy - L, sx + 1, yy, C(0.35, 0.38, 0.48), 0.28)
        rect(view, sx, yy, sx + 1, yy + 1, C(0.7, 0.72, 0.8), 0.55)
    wmask = (XX >= wx0) & (XX < wx1) & (YY >= 20) & (YY < 205)
    img = np.where(wmask[..., None], view, img)
    wall = C(0.034, 0.031, 0.035)
    img[(XX < wx0) & (YY < 215)] = wall
    for px in range(0, 330, 55):
        rect(img, px - camx, 20, px - camx + 1, 215, C(0.052, 0.047, 0.052))
    rect(img, -camx, 0, IW - camx, 20, C(0.018, 0.018, 0.022))
    for px in range(40, IW, 90):
        rect(img, px - camx, 18, px - camx + 8, 20, C(0.45, 0.35, 0.22))
        glow(img, px - camx + 4, 26, 18, C(0.22, 0.16, 0.09), 0.35)
    for mx in range(330, IW + 1, 86):
        rect(img, mx - camx - 1, 20, mx - camx + 2, 205, C(0.012, 0.012, 0.016))
    rect(img, wx0, 18, wx1, 21, C(0.012, 0.012, 0.016))
    rect(img, wx0, 203, wx1, 207, C(0.06, 0.055, 0.06))
    # painting
    px0, py0 = int(95 - camx), 42
    rect(img, px0 - 5, py0 - 5, px0 + PAINT_W + 5, py0 + PAINT_H + 5, C(0.16, 0.12, 0.06))
    rect(img, px0 - 4, py0 - 4, px0 + PAINT_W + 4, py0 + PAINT_H + 4, C(0.09, 0.07, 0.035))
    rect(img, px0 - 2, py0 - 2, px0 + PAINT_W + 2, py0 + PAINT_H + 2, C(0.03, 0.025, 0.02))
    xs0, xs1 = max(0, px0), min(W, px0 + PAINT_W)
    if xs1 > xs0:
        img[py0:py0 + PAINT_H, xs0:xs1] = PAINT[:, xs0 - px0:xs1 - px0] * (1 + fl * 1.2)
    rect(img, px0 + 40, py0 - 11, px0 + 80, py0 - 8, C(0.2, 0.17, 0.11))
    glow(img, px0 + 60, py0 + 8, 75, C(0.20, 0.11, 0.05), 0.5)
    eg = 0.5 + 0.5 * math.sin(t * 3)
    for ex in (0.5 - 0.27 * 0.46 * PAINT_H / PAINT_W, 0.5 + 0.27 * 0.46 * PAINT_H / PAINT_W):
        glow(img, px0 + PAINT_W * ex, py0 + PAINT_H * (0.5 - 0.08 * 0.46), 7, C(0.8, 0.15, 0.03), 0.2 + 0.2 * eg)
    # floor
    fy = 215
    img[fy:] = C(0.026, 0.024, 0.028)
    for x in range(330, IW, 86):
        for d in range(0, 55):
            a = 0.22 * (1 - d / 55) ** 1.4
            rect(img, x - camx + 3, fy + d, x - camx + 83, fy + d + 1, C(0.16, 0.16, 0.24) + fl * 0.5, a)
    rect(img, -camx, fy - 1, IW - camx, fy, C(0.07, 0.06, 0.06))
    # rug
    rx, ry = 185 - camx, 245
    rug = ellipse_mask(rx, ry, 70, 10)
    for sgn in (-1, 1):
        rug |= poly_mask([(rx + sgn * 38, ry - 5), (rx + sgn * 58, ry - 15), (rx + sgn * 66, ry - 13), (rx + sgn * 50, ry - 2)])
        rug |= poly_mask([(rx + sgn * 38, ry + 5), (rx + sgn * 60, ry + 16), (rx + sgn * 68, ry + 14), (rx + sgn * 50, ry + 2)])
    head = ellipse_mask(rx - 80, ry - 1, 12, 8)
    rug |= head | ellipse_mask(rx - 86, ry - 8, 3, 3) | ellipse_mask(rx - 86, ry + 6, 3, 3)
    rug |= (np.abs(YY - (ry + np.sin((XX - rx) / 8) * 3)) < 1.2) & (XX > rx + 68) & (XX < rx + 96)
    blend(img, rug.astype(np.float32), C(0.068, 0.062, 0.066))
    stripes = rug & (np.sin((XX - rx) * 0.30 + np.sin((YY - ry) * 0.5) * 1.6) > 0.72) & ~head
    blend(img, stripes.astype(np.float32), C(0.035, 0.032, 0.036))
    blend(img, edge_rim(rug, 0, -1).astype(np.float32), C(0.11, 0.10, 0.11))
    for ey in (-3, 2):
        rect(img, rx - 86, ry + ey, rx - 84, ry + ey + 1, C(0.75, 0.1, 0.05))
    # sofa
    sx0 = 60 - camx
    so = C(0.042, 0.042, 0.052)
    rect(img, sx0, 158, sx0 + 220, 200, so)
    rect(img, sx0 - 8, 176, sx0 + 228, 214, so * 1.2)
    rect(img, sx0 - 12, 168, sx0 + 4, 214, so * 0.9)
    rect(img, sx0 + 216, 168, sx0 + 232, 214, so * 0.9)
    for k in range(3):
        rect(img, sx0 + 5 + k * 72, 177, sx0 + 6 + k * 72, 196, C(0.02, 0.02, 0.025))
    rect(img, sx0, 158, sx0 + 220, 159, C(0.10, 0.095, 0.11))
    rect(img, sx0 - 8, 176, sx0 + 228, 177, C(0.11, 0.10, 0.115))
    rect(img, sx0 - 12, 168, sx0 + 4, 169, C(0.09, 0.085, 0.1)); rect(img, sx0 + 216, 168, sx0 + 232, 169, C(0.09, 0.085, 0.1))
    # boss + lighter
    S = 1.5
    bx = 598 - camx; by = 100
    flame = 1.0 - 0.3 * ((frame * 7) % 3) / 2 if 9.95 <= t < 10.38 else 0.0
    light = 0.62 + flame * 0.45
    boss_small(img, bx, by, light, fl, S)
    cx_, cy_ = bx + 6.5 * S, by + 15.5 * S
    if flame > 0:
        glow(img, cx_, cy_ - 3, 22, C(1.0, 0.55, 0.15), 0.35 * flame)
        rect(img, cx_ - 1, cy_ - 7, cx_ + 1, cy_ - 2, C(1, 0.85, 0.4))
        rect(img, cx_ - 0.5, cy_ - 9, cx_ + 0.5, cy_ - 7, C(1, 0.6, 0.2))
        rect(img, cx_ - 2, cy_ - 2, cx_ + 2, cy_ + 4, C(0.62, 0.62, 0.66))
        rect(img, cx_ - 2, cy_ - 2, cx_ + 2, cy_ - 1, C(0.85, 0.85, 0.9))
    if t >= 10.15:
        rect(img, cx_, cy_ - 1, cx_ + 1, cy_ + 1, C(1, 0.4, 0.1))
        glow(img, cx_ + 0.5, cy_, 6, C(1, 0.3, 0.05), 0.4)
    # desk
    dx0, dx1 = 480 - camx, 700 - camx
    rect(img, dx0, 176, dx1, 180, C(0.09, 0.07, 0.055))
    rect(img, dx0, 175, dx1, 176, C(0.2, 0.15, 0.1))
    rect(img, dx0 + 6, 180, dx1 - 6, 226, C(0.038, 0.029, 0.026))
    for k in range(4):
        rect(img, dx0 + 6 + k * 52, 180, dx0 + 7 + k * 52, 226, C(0.065, 0.048, 0.04))
    lx = 515 - camx
    rect(img, lx - 1, 160, lx + 1, 175, C(0.12, 0.10, 0.06))
    rect(img, lx - 6, 172, lx + 6, 175, C(0.12, 0.10, 0.06))
    shade = poly_mask([(lx - 5, 150), (lx + 5, 150), (lx + 9, 161), (lx - 9, 161)])
    blend(img, shade.astype(np.float32), C(0.72, 0.40, 0.14))
    rect(img, lx - 9, 160, lx + 9, 161, C(1, 0.8, 0.45))
    glow(img, lx, 168, 60, C(0.55, 0.30, 0.09), 0.55)
    ax = 655 - camx
    rect(img, ax - 7, 172, ax + 7, 175, C(0.3, 0.3, 0.33)); rect(img, ax - 5, 172, ax + 5, 173, C(0.08, 0.08, 0.09))
    rect(img, 540 - camx, 173, 565 - camx, 175, C(0.5, 0.47, 0.41))
    rect(img, 630 - camx, 164, 636 - camx, 175, C(0.35, 0.22, 0.1), 0.8)
    rect(img, 630 - camx, 164, 636 - camx, 165, C(0.7, 0.6, 0.45))
    # foreground pillar (parallax)
    fpx = 430 - camx * 1.6
    rect(img, fpx, 0, fpx + 34, H, C(0.008, 0.008, 0.01))
    rect(img, fpx + 33, 0, fpx + 34, H, C(0.05, 0.05, 0.065))
    vg = np.clip(1 - (((XX - W / 2) / (W * 0.62)) ** 2 + ((YY - H / 2) / (H * 0.75)) ** 2), 0, 1) ** 0.6
    img *= (0.35 + 0.65 * vg)[..., None]
    img += fl * 0.12
    return img

# ---------------------------------------------------------------- close-up
PS = 280
_ref = Image.open('ref_clean.png').convert('RGB').resize((PS, PS), Image.LANCZOS)
_q = _ref.quantize(28, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
PORT = np.asarray(_q.convert('RGB'), np.float32) / 255
SC = PS / 1200
EMBER = (1128 * SC, 962 * SC)
MOUTH = (800 * SC, 735 * SC)
pys, pxs = np.mgrid[0:PS, 0:PS].astype(np.float32)
CANDLE = (pxs < 250 * SC) & (pys > 300 * SC) & (pys < 480 * SC) & (PORT[..., 0] > 0.45)

class Smoke:
    def __init__(self):
        self.p = []; self.r = np.random.default_rng(99)
    def emit(self, x, y, vx, vy, n, s0, life):
        for _ in range(n):
            self.p.append([x + self.r.normal() * 1.2, y + self.r.normal() * 1.2, vx + self.r.normal() * 0.25, vy + self.r.normal() * 0.2, s0, 0.0, life, self.r.random() * 6])
    def step(self, dt):
        acc = np.zeros((H, W), np.float32)
        keep = []
        for q in self.p:
            q[0] += q[2]; q[1] += q[3]; q[2] *= 0.975; q[3] = q[3] * 0.98 - 0.018
            q[2] += math.sin(q[1] * 0.07 + q[7]) * 0.06
            q[5] += dt
            if q[5] < q[6]:
                keep.append(q)
                a = (1 - q[5] / q[6]) * 0.25
                r = q[4] + q[5] * 8
                x0, x1 = int(max(0, q[0] - 2 * r)), int(min(W, q[0] + 2 * r + 1))
                y0, y1 = int(max(0, q[1] - 2 * r)), int(min(H, q[1] + 2 * r + 1))
                if x1 > x0 and y1 > y0:
                    acc[y0:y1, x0:x1] += a * np.exp(-((XX[y0:y1, x0:x1] - q[0]) ** 2 + (YY[y0:y1, x0:x1] - q[1]) ** 2) / (r * r))
        self.p = keep
        return acc

SMOKE = Smoke()
_r2 = np.random.default_rng(77)
CL_DRIPS = [(_r2.random() * 230, _r2.random() * 270, 4 + int(_r2.integers(0, 10)), 0.4 + _r2.random()) for _ in range(45)]
CL_BOKEH = [(_r2.random() * 220, _r2.random() * 250, int(_r2.integers(0, 4)), 4 + _r2.random() * 9) for _ in range(20)]
SMOKE_T = [-1]

def closeup(t, frame):
    u = t - 10.45
    img = np.zeros((H, W, 3), np.float32)
    fl = flash_at(t)
    img[:] = C(0.012, 0.012, 0.022)
    for (x, y, k, r) in CL_BOKEH:
        col = [C(0.9, 0.55, 0.2), C(0.35, 0.45, 0.8), C(0.8, 0.12, 0.08), C(0.6, 0.6, 0.55)][k]
        m = ellipse_mask(x + u * 2, y, r, r).astype(np.float32)
        blend(img, m, col, 0.07)
        blend(img, edge_rim(m > 0, 0, -1).astype(np.float32), col, 0.08)
    for (x, y, L, sp) in CL_DRIPS:
        yy = (y + sp * frame) % 280
        rect(img, x, yy - L, x + 1, yy, C(0.22, 0.24, 0.32), 0.35)
        rect(img, x, yy, x + 1, yy + 1, C(0.45, 0.47, 0.58), 0.55)
    img += fl * C(0.22, 0.24, 0.34) * np.clip(1 - XX / 260, 0, 1)[..., None]
    if u < 0.9:
        e = 0.55 + 0.45 * ease_io(u / 0.9) + 0.08 * math.sin(frame * 2.3)
    elif u < 1.2:
        e = lerp(1.0, 0.55, (u - 0.9) / 0.3)
    else:
        e = 0.5 + 0.06 * math.sin(frame * 1.7)
    amb = lerp(0.28, 0.80, ease_io(u / 1.1))
    amb = round(amb * 10) / 10
    lx, ly = EMBER
    d2 = ((pxs - lx) ** 2 + (pys - ly) ** 2) / (95 ** 2)
    warm = np.exp(-d2) * e
    port = PORT * (amb + warm * 0.5)[..., None] * C(1.0, 0.94, 0.87)
    port += PORT * (warm * 0.35)[..., None] * C(0.4, 0.12, 0.0)
    fk = 0.8 + 0.25 * math.sin(frame * 1.9) * math.sin(frame * 0.7)
    port[CANDLE] *= fk
    port += fl * (PORT * 0.55 + 0.03) * C(0.55, 0.62, 0.85)
    ox = int(round(W - PS + 4 - ease_io(u / 3.2) * 6)); oy = int(round(-8 + ease_io(u / 3.2) * 6))
    fade = np.clip((pxs - 15) / 95, 0, 1) ** 1.2
    fade = np.floor(fade * 6) / 6
    y0, y1 = max(0, oy), min(H, oy + PS); x0, x1 = max(0, ox), min(W, ox + PS)
    sub = port[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    a = fade[y0 - oy:y1 - oy, x0 - ox:x1 - ox][..., None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + sub * a
    ex, ey = ox + lx, oy + ly
    glow(img, ex, ey, 22, C(1.0, 0.35, 0.05), 0.55 * e)
    rect(img, ex - 2, ey - 1, ex + 2, ey + 2, C(1.0, 0.45 + 0.3 * e, 0.12))
    rect(img, ex - 1, ey, ex + 1, ey + 1, C(1.0, 0.9, 0.6))
    if SMOKE_T[0] != frame:
        SMOKE_T[0] = frame
        if frame % 2 == 0:
            SMOKE.emit(ex - 1, ey - 3, -0.15, -0.55, 1, 1.4, 3.0)
        if 1.25 <= u < 2.0:
            k = (u - 1.25) / 0.75
            SMOKE.emit(ox + MOUTH[0] - 10, oy + MOUTH[1] + 3, -2.6 * (1 - k) - 0.8, -0.25 - 0.3 * k, 2, 1.6, 2.4)
        SMOKE.m = SMOKE.step(1 / FPS)
    m = np.floor(np.clip(SMOKE.m * 1.4, 0, 0.99) * 4) / 4
    smoke_col = C(0.55, 0.55, 0.62) * (0.55 + 0.45 * amb) + fl * 0.4
    blend(img, m, smoke_col, 0.45)
    vg = np.clip(1 - (((XX - W * 0.6) / (W * 0.7)) ** 2 + ((YY - H / 2) / (H * 0.8)) ** 2), 0, 1) ** 0.7
    img *= (0.3 + 0.7 * vg)[..., None]
    fo = 1 - ease_io(seg(t, 13.05, 13.75))
    img = img * fo
    ef = 1 - ease_io(seg(t, 13.55, 13.95))
    if fo < 1:
        glow(img, ex, ey, 14, C(1.0, 0.3, 0.05), 0.5 * ef)
        rect(img, ex - 1, ey, ex + 1, ey + 1, C(1.0, 0.45, 0.1) * ef)
    return img

# ---------------------------------------------------------------- white flash + main
def render(t, frame):
    if t < 7.47:
        img = exterior(t, frame)
    elif t < 10.45:
        img = interior(t, frame)
    else:
        img = closeup(t, frame)
    # transition white
    w = 0.0
    if 7.38 <= t < 7.47: w = (t - 7.38) / 0.09
    elif 7.47 <= t < 7.95: w = (1 - (t - 7.47) / 0.48) ** 2
    if w > 0:
        img = img * (1 - w) + C(0.92, 0.94, 1.0) * w
    # hard cut dip to black between interior and closeup
    if 10.40 <= t < 10.50:
        img = img * 0.2
    return np.clip(img, 0, 1)

if __name__ == '__main__':
    import os
    out = sys.argv[1] if len(sys.argv) > 1 else 'frames.npy'
    only = [float(a) for a in sys.argv[2:]]
    if only:
        for tt in only:
            fr = int(tt * FPS)
            # warm up smoke state for closeup previews
            if tt >= 10.45:
                for f0 in range(int(10.45 * FPS), fr):
                    closeup(f0 / FPS, f0)
            im = render(tt, fr)
            Image.fromarray((im * 255).astype(np.uint8)).resize((W * 2, H * 2), Image.NEAREST).save(f'prev_{tt:05.2f}.png')
            print('prev', tt)
    else:
        arr = np.zeros((NF, H, W, 3), np.uint8)
        for f in range(NF):
            arr[f] = (render(f / FPS, f) * 255 + 0.5).astype(np.uint8)
            if f % 24 == 0: print('frame', f, flush=True)
        np.save(out, arr)
