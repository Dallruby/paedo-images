import numpy as np, math
from PIL import Image
def C(*v): return np.array(v,np.float32)
def pin(u,v,pts):
    pts=np.asarray(pts,np.float32); ins=np.zeros(u.shape,bool); j=len(pts)-1
    for i in range(len(pts)):
        xi,yi=pts[i]; xj,yj=pts[j]
        ins^=((yi>v)!=(yj>v))&(u<(xj-xi)*(v-yi)/(yj-yi+1e-9)+xi); j=i
    return ins
def rell(u,v,cx,cy,rx,ry,ang):
    c,s=math.cos(ang),math.sin(ang); x=u-cx; y=v-cy
    xr=x*c+y*s; yr=-x*s+y*c
    return (xr/rx)**2+(yr/ry)**2<=1
def tiger(w,h,seed=9):
    ys,xs=np.mgrid[0:h,0:w].astype(np.float32)
    u=(xs+0.5-w/2)/(h*0.46); v=(ys+0.5-h/2)/(h*0.46)
    r=np.random.default_rng(seed)
    g=r.random((h//6+2,w//6+2)).astype(np.float32)
    bgn=np.asarray(Image.fromarray((g*255).astype(np.uint8)).resize((w+12,h+12),Image.BICUBIC),np.float32)[6:6+h,6:6+w]/255
    img=C(0.20,0.045,0.035)*(0.7+0.5*bgn)[...,None]
    # vignette of the canvas
    img*=np.clip(1.25-0.35*(u**2+v**2),0.4,1.2)[...,None]
    au=np.abs(u)
    ang=np.arctan2(v,u)
    tuft=0.05*np.sin(ang*14)
    face=pin(u,v,[(-0.5,-0.62),(-0.78,-0.25),(-0.95,0.2),(-0.82,0.55),(-0.5,0.85),(-0.2,0.98),(0.2,0.98),(0.5,0.85),(0.82,0.55),(0.95,0.2),(0.78,-0.25),(0.5,-0.62),(0.25,-0.72),(-0.25,-0.72)])
    ruff=(((u/1.0)**2+((v-0.2)/0.85)**2)<=(1+tuft)) & (v>-0.3)
    ears=rell(au,v,0.60,-0.70,0.2,0.2,0)
    earin=rell(au,v,0.60,-0.68,0.10,0.11,0)
    m=face|ruff|ears
    fur=C(0.085,0.08,0.085)*(0.85+0.4*bgn)[...,None]
    img=np.where(m[...,None],fur,img)
    sheen=m&(v<-0.1)&(((u/0.7)**2+((v+0.35)/0.35)**2)<=1)
    img=np.where(sheen[...,None],img*1.45,img)
    img=np.where(earin[...,None],C(0.02,0.015,0.02),img)
    # light fur: brows, cheeks under eyes, muzzle
    light=C(0.36,0.34,0.33)
    brow=rell(au,v,0.28,-0.28,0.13,0.05,-0.35)
    cheek=rell(au,v,0.30,0.10,0.09,0.045,0.3)
    muz=rell(au,v,0.13,0.46,0.15,0.13,0)
    chin=rell(u,v,0,0.72,0.16,0.12,0)
    img=np.where((brow|cheek|muz|chin)[...,None],light,img)
    # stripes
    st=np.zeros(u.shape,bool)
    for vb,wb in [(-0.58,0.20),(-0.48,0.15),(-0.38,0.10)]:
        st|=(np.abs(v-vb)<0.028)&(au<wb)
    st|=(au<0.025)&(v>-0.6)&(v<-0.2)
    for v0 in (-0.05,0.22,0.48,0.70):
        t=np.clip((au-0.5)/0.45,0,1)
        st|=(np.abs(v-(v0-0.18*(au-0.55)))<0.05*t)&(au>0.5)
    for u0 in (0.32,0.44):
        st|=(np.abs(au-u0-(v+0.62)*0.25)<0.022)&(v<-0.35)&(v>-0.66)
    st&=m&~(muz|chin)
    img=np.where(st[...,None],C(0.012,0.01,0.012),img)
    # brow ridge (angry)
    ridge=rell(au,v,0.26,-0.20,0.17,0.035,-0.45)
    img=np.where(ridge[...,None],C(0.01,0.01,0.01),img)
    # eyes
    eye=rell(au,v,0.27,-0.08,0.12,0.05,-0.30)
    img=np.where(eye[...,None],C(1.0,0.28,0.06),img)
    eyeg=rell(au,v,0.27,-0.085,0.07,0.028,-0.30)
    img=np.where(eyeg[...,None],C(1.0,0.75,0.25),img)
    pup=(np.abs(au-0.27)<0.018)&eye
    img=np.where(pup[...,None],C(0.02,0,0),img)
    eo=rell(au,v,0.27,-0.08,0.14,0.07,-0.30)&~eye
    img=np.where(eo[...,None],C(0.0,0.0,0.0),img)
    # nose
    nose=pin(u,v,[(-0.14,0.24),(0.14,0.24),(0.02,0.38),(-0.02,0.38)])
    img=np.where(nose[...,None],C(0.12,0.04,0.04),img)
    mouth=((np.abs(u)<0.012)&(v>0.38)&(v<0.5))|(np.abs(v-(0.5+0.25*(au-0.0)**1.0))<0.018)&(au<0.2)
    img=np.where(mouth[...,None],C(0.01,0.0,0.0),img)
    fang=pin(au,v,[(0.08,0.52),(0.13,0.54),(0.10,0.66)])
    img=np.where(fang[...,None],C(0.85,0.82,0.75),img)
    # outline
    e=m&~(np.roll(m,1,0)&np.roll(m,-1,0)&np.roll(m,1,1)&np.roll(m,-1,1))
    img=np.where(e[...,None],C(0,0,0),img)
    return img
if __name__=='__main__':
    im=tiger(120,92)
    Image.fromarray((np.clip(im,0,1)*255).astype(np.uint8)).resize((480,368),Image.NEAREST).save('tiger.png')
