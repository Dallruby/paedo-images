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
