"""
digitize_valdyn_plots.py
Re-creates reference_valdyn_digitized.npz from the screen photos embedded in
Benchmark-Daten-Kalibrierung.key (a Keynote file is a zip archive).

Method (per plot):
 1. approximate plot-frame corners (picked by hand in a 1800 px preview)
 2. refine the four frame edges by searching the darkest line nearby and
    fitting straight lines -> exact corners
 3. homography (perspective transform) photo pixels -> data coordinates
 4. colour mask (red/green/blue/cyan) -> point cloud of the curve
 5. de-duplicate on a 1200x500 grid
Usage:  python digitize_valdyn_plots.py ../Benchmark-Daten-Kalibrierung.key
"""
import sys, zipfile, io, pickle

import numpy as np, json, sys
from PIL import Image, ImageOps
S = 5712/1800.0

def load(path):
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGB')).astype(np.int16)

def refine_edge(gray, p0, p1, vertical, search=25, thr=None):
    """Fit frame line between approx points p0,p1 (orig px) by locating darkest pixel across the line."""
    n = 60; pts = []
    for t in np.linspace(0.12, 0.88, n):
        x = p0[0]+(p1[0]-p0[0])*t; y = p0[1]+(p1[1]-p0[1])*t
        if vertical:
            xs = np.arange(int(x-search), int(x+search)); prof = gray[int(y), xs]
            k = np.argmin(prof); pts.append((xs[k], y))
        else:
            ys = np.arange(int(y-search), int(y+search)); prof = gray[ys, int(x)]
            k = np.argmin(prof); pts.append((x, ys[k]))
    pts = np.array(pts, float)
    if vertical:  # x = a*y + b  (robust: iterate)
        A = np.c_[pts[:,1], np.ones(len(pts))]; v = pts[:,0]
    else:
        A = np.c_[pts[:,0], np.ones(len(pts))]; v = pts[:,1]
    m = np.ones(len(v), bool)
    for _ in range(3):
        c, *_ = np.linalg.lstsq(A[m], v[m], rcond=None)
        r = np.abs(A@c - v); m = r < max(2.0, 2.5*np.median(r[m]))
    return c

def intersect(cv, ch):
    # x = a y + b ; y = c x + d
    a,b = cv; c,d = ch
    y = (c*b + d)/(1 - c*a); x = a*y + b
    return np.array([x, y])

def homography(src, dst):
    A=[]
    for (x,y),(u,v) in zip(src,dst):
        A.append([x,y,1,0,0,0,-u*x,-u*y,-u]); A.append([0,0,0,x,y,1,-v*x,-v*y,-v])
    _,_,Vt=np.linalg.svd(np.array(A,float)); H=Vt[-1].reshape(3,3); return H/H[2,2]

def apply(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T; return p[:,:2]/p[:,2:3]

def digitize(img, corners1800, xr, yr, color, refine=True, margin=4):
    """corners1800: TL,TR,BR,BL approx in 1800-px preview coords. xr=(xmin,xmax) yr=(ymin,ymax) at frame."""
    gray = img.mean(axis=2)
    c = np.array(corners1800, float)*S
    TL,TR,BR,BL = c
    if refine:
        L = refine_edge(gray, TL, BL, True); R = refine_edge(gray, TR, BR, True)
        T = refine_edge(gray, TL, TR, False); B = refine_edge(gray, BL, BR, False)
        TL,TR,BR,BL = intersect(L,T), intersect(R,T), intersect(R,B), intersect(L,B)
    src = np.array([TL,TR,BR,BL]); dst = np.array([[xr[0],yr[1]],[xr[1],yr[1]],[xr[1],yr[0]],[xr[0],yr[0]]])
    H = homography(src, dst)
    x0=int(min(TL[0],BL[0]))+margin; x1=int(max(TR[0],BR[0]))-margin
    y0=int(min(TL[1],TR[1]))+margin; y1=int(max(BL[1],BR[1]))-margin
    sub = img[y0:y1, x0:x1]; R_,G_,B_ = sub[...,0],sub[...,1],sub[...,2]
    if color=='red':   m = (R_-G_>45)&(R_-B_>30)
    elif color=='green': m = (G_-R_>25)&(G_-B_>15)
    elif color=='blue':  m = (B_-R_>40)&(B_-G_>15)
    elif color=='cyan':  m = (G_-R_>30)&(B_-R_>30)
    else: raise ValueError
    yy,xx = np.nonzero(m)
    pts = apply(H, np.c_[xx+x0, yy+y0].astype(float))
    return pts, src

def envelope(pts, xr, n=800):
    xg = np.linspace(xr[0], xr[1], n+1); xc = 0.5*(xg[1:]+xg[:-1])
    lo = np.full(n, np.nan); hi = np.full(n, np.nan); md = np.full(n, np.nan)
    idx = np.clip(np.digitize(pts[:,0], xg)-1, 0, n-1)
    for i in range(n):
        v = pts[idx==i,1]
        if len(v)>=2: lo[i]=np.percentile(v,3); hi[i]=np.percentile(v,97); md[i]=np.median(v)
    return xc, lo, md, hi


specs = {
 # name: (file, corners TL,TR,BR,BL in 1800px, xr, yr, colors)
 's1_lashA': ('IMG_3834-9130.jpeg', [(150,582),(776,582),(776,1003),(150,1002)], (0,400), (-0.05,0.30), ['red']),
 's1_lashB': ('IMG_3834-9130.jpeg', [(1007,585),(1622,588),(1622,1003),(1007,1003)], (0,400), (-0.05,0.30), ['red']),
 's1_FA':    ('IMG_3833-9124.jpeg', [(163,650),(810,654),(808,1105),(158,1102)], (0,400), (-2000,10000), ['green']),
 's1_FB':    ('IMG_3833-9124.jpeg', [(1068,656),(1717,659),(1717,1106),(1068,1106)], (0,400), (-2000,10000), ['green']),
 's1_XM1':   ('IMG_3835-9135.jpeg', [(330,490),(838,493),(838,792),(330,790)], (0,400), (-2000,10000), ['red']),
 's1_YM1':   ('IMG_3835-9135.jpeg', [(1032,497),(1527,500),(1527,792),(1030,790)], (0,400), (-4e-3,4e-3), ['green']),
 's1_XM2':   ('IMG_3835-9135.jpeg', [(337,905),(836,905),(836,1200),(340,1200)], (0,400), (131.5595,131.5625), ['red']),
 's1_YM2':   ('IMG_3835-9135.jpeg', [(1030,902),(1522,903),(1522,1197),(1030,1195)], (0,400), (47.876,47.884), ['green']),
 's2_T':     ('IMG_3837-9164.jpeg', [(316,311),(855,313),(855,673),(316,673)], (0,1500), (-10,10), ['red']),
 's2_Ft':    ('IMG_3837-9164.jpeg', [(1050,316),(1572,322),(1572,677),(1050,675)], (0,1500), (-1000,1000), ['red']),
 's2_lash':  ('IMG_3837-9164.jpeg', [(318,812),(855,812),(855,1170),(320,1170)], (0,1500), (-0.02,0.06), ['red','blue']),
 's2_X':     ('IMG_3838-9169.jpeg', [(258,333),(826,333),(830,1100),(268,1100)], (720,1440), (131.52,131.60), ['red','blue','cyan']),
 's2_Y':     ('IMG_3838-9169.jpeg', [(1033,330),(1590,330),(1590,1082),(1035,1092)], (720,1440), (47.870,47.890), ['red','blue','cyan']),
 's2_FX':    ('IMG_3839-9174.jpeg', [(290,368),(893,372),(890,1177),(290,1173)], (720,1440), (-600,400), ['red','blue']),
 's2_FY':    ('IMG_3839-9174.jpeg', [(1113,372),(1708,375),(1705,1180),(1113,1180)], (720,1440), (-1000,1000), ['red','blue']),
}


if __name__ == '__main__':
    key = sys.argv[1] if len(sys.argv) > 1 else '../Benchmark-Daten-Kalibrierung.key'
    z = zipfile.ZipFile(key)
    names = {n.split('/')[-1]: n for n in z.namelist()}
    out = {}
    cache = {}
    for name, (f, c, xr, yr, cols) in specs.items():
        if f not in cache:
            cache[f] = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(z.read(names[f])))).convert('RGB')).astype(np.int16)
        for col in cols:
            p, _ = digitize(cache[f], c, xr, yr, col)
            ix = np.round((p[:, 0]-xr[0])/(xr[1]-xr[0])*1200).astype(int); iy = np.round((p[:, 1]-yr[0])/(yr[1]-yr[0])*500).astype(int)
            _, u = np.unique(ix*1000+iy, return_index=True); p = p[np.sort(u)]; p = p[np.argsort(p[:, 0])]
            k = f'{name}_{col}'
            out[k] = p; out[k+'_xr'] = np.array(xr); out[k+'_yr'] = np.array(yr)
            print(k, len(p))
    np.savez_compressed('reference_valdyn_digitized.npz', **out)
