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

