import numpy as np
from PIL import Image, ImageFilter

def srgb2lab(c):
    c=np.asarray(c,np.float32)/255
    c=np.where(c>0.04045,((c+0.055)/1.055)**2.4,c/12.92)
    M=np.array([[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]],np.float32)
    xyz=c@M.T/np.array([0.9505,1.0,1.089],np.float32)
    f=np.where(xyz>0.008856,np.cbrt(xyz),7.787*xyz+16/116)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],-1)

def kmeans_lab(pix,k,iters=25,seed=0,init=None):
    lab=srgb2lab(pix)
    r=np.random.default_rng(seed)
    if init is None:
        # kmeans++ init
        cent=[lab[r.integers(len(lab))]]
        for _ in range(k-1):
            d=np.min(((lab[:,None]-np.array(cent)[None])**2).sum(-1),1)
            cent.append(lab[r.choice(len(lab),p=d/d.sum())])
        cent=np.array(cent)
    else: cent=srgb2lab(init)
    for _ in range(iters):
        a=((lab[:,None]-cent[None])**2).sum(-1).argmin(1)
        for j in range(k):
            m=a==j
            if m.any(): cent[j]=lab[m].mean(0)
    # return rgb centroid (mean of members in rgb)
    out=np.zeros((k,3),np.float32)
    for j in range(k):
        m=a==j
        out[j]=pix[m].mean(0) if m.any() else 0
    return out

def quantize(img,pal):
    lab=srgb2lab(img.reshape(-1,3)); pl=srgb2lab(pal)
    idx=((lab[:,None]-pl[None])**2).sum(-1).argmin(1)
    return idx.reshape(img.shape[:2])

def clean_orphans(idx,protect=None,passes=2):
    H,W=idx.shape
    for _ in range(passes):
        p=np.pad(idx,1,mode='edge')
        nb=np.stack([p[1+dy:H+1+dy,1+dx:W+1+dx] for dy in (-1,0,1) for dx in (-1,0,1) if (dy,dx)!=(0,0)],0)
        same=(nb==idx[None]).sum(0)
        orphan=same==0
        if protect is not None: orphan&=~protect
        ys,xs=np.nonzero(orphan)
        for y,x in zip(ys,xs):
            v,c=np.unique(nb[:,y,x],return_counts=True)
            idx[y,x]=v[c.argmax()]
    return idx
