import numpy as np
from PIL import Image
from pxlib import kmeans_lab, quantize, clean_orphans
W = 384
SETS = {'A': ['1'], 'B': ['2'], 'C1': ['3-1'], 'C2': ['3-2'], 'D': ['4-1', '4-2', '4-3'], 'E': ['5-1', '5-2', '5-4'], 'F': ['6-1', '6-2', '6-3', '6-4', '6-5']}
NCOL = {'A': 40, 'B': 32, 'C1': 40, 'C2': 36, 'D': 40, 'E': 32, 'F': 40}
out = {}
for s, names in SETS.items():
    los = []
    for n in names:
        im = Image.open(f'src/{n}.png').convert('RGB')
        w, h = im.size
        nh = int(round(h * W / w))
        # median-ish: downscale by area average (source pixels ~4px)
        lo = np.asarray(im.resize((W, nh), Image.BOX), np.float32)
        los.append(lo)
    pix = np.concatenate([l.reshape(-1, 3) for l in los])[::3]
    pal = kmeans_lab(pix, NCOL[s], iters=18)
    for n, lo in zip(names, los):
        idx = quantize(lo, pal)
        idx = clean_orphans(idx, None, 1)
        np.save(f'n_{n}.npy', idx)
        Image.fromarray(pal[idx].astype(np.uint8)).save(f'n_{n}.png')
    np.save(f'pal_{s}.npy', pal)
    print(s, 'done')
