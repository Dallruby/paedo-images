import numpy as np
from PIL import Image, ImageFilter
im=np.asarray(Image.open('ref.png').convert('RGB')).astype(np.float32)
x0,x1,y0,y1=270,385,740,1040
reg=im[y0:y1,x0:x1].copy()
lum=reg.mean(2)
m=Image.fromarray(((lum>55)*255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))
mask=np.asarray(m)>0
fill=reg.copy()
fill[mask]=0
w=(~mask).astype(np.float32)
acc=fill*w[...,None]
for i in range(400):
    # diffusion
    a=np.pad(acc,((1,1),(1,1),(0,0)),mode='edge'); ww=np.pad(w,1,mode='edge')
    s=(a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:]); sw=(ww[:-2,1:-1]+ww[2:,1:-1]+ww[1:-1,:-2]+ww[1:-1,2:])
    nv=s/np.maximum(sw,1e-6)[...,None]
    upd=mask&(sw>0)
    acc[upd]=nv[upd]; w=np.where(upd,1.0,w).astype(np.float32)
    acc[~mask]=reg[~mask]
reg2=acc
im[y0:y1,x0:x1]=reg2
Image.fromarray(im.clip(0,255).astype(np.uint8)).save('ref_clean.png')
Image.fromarray(im[700:1080,220:420].clip(0,255).astype(np.uint8)).save('chk.png')
