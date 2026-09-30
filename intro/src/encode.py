import numpy as np, subprocess, imageio_ffmpeg
from PIL import Image
arr=np.load('frames.npy')
NF,H,W,_=arr.shape
rng=np.random.default_rng(0)
idx=rng.integers(0,NF*H*W,300000)
flat=arr.reshape(-1,3)[idx]
mos=Image.fromarray(flat.reshape(500,600,3))
pal=np.asarray(mos.quantize(80,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE).convert('RGB').getpalette()[:240] if False else None)
q=mos.quantize(80,method=Image.Quantize.MAXCOVERAGE,dither=Image.Dither.NONE)
pal=np.array(q.getpalette()[:80*3],np.float32).reshape(-1,3)
# make sure pure black-ish and ember colors exist
extra=np.array([[0,0,0],[255,240,200],[255,120,30],[230,25,15],[140,185,215]],np.float32)
pal=np.concatenate([pal,extra])
# 6-bit LUT
g=np.arange(64)*4+2
cube=np.stack(np.meshgrid(g,g,g,indexing='ij'),-1).reshape(-1,3).astype(np.float32)
lut=np.zeros(len(cube),np.int32)
for i in range(0,len(cube),16384):
    d=((cube[i:i+16384,None,:]-pal[None])**2*np.array([0.3,0.59,0.11])).sum(-1)
    lut[i:i+16384]=d.argmin(1)
pal8=pal.astype(np.uint8)
np.save('palette.npy',pal8)
ff=imageio_ffmpeg.get_ffmpeg_exe()
S=4
p=subprocess.Popen([ff,'-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W*S}x{H*S}','-r','24','-i','-','-i','intro.wav',
  '-c:v','libx264','-preset','slow','-crf','14','-tune','animation','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart','paedo_intro.mp4'],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
qs=[]
for f in range(NF):
    a=arr[f]>>2
    li=(a[...,0].astype(np.int32)*64+a[...,1])*64+a[...,2]
    qf=arr[f]
    qs.append(qf)
    big=qf.repeat(S,0).repeat(S,1)
    p.stdin.write(big.tobytes())
p.stdin.close(); p.wait()
np.save('frames_q.npy',np.stack(qs))
print('done',p.returncode)
