import numpy as np, sys
sys.path.insert(0,'.')
from mapdata import load, TILE
from sklearn.metrics import average_precision_score as AP
md=load(); times=np.array(md["times"]); fire=md["fire"]; lsm=md["lsm"]
print(times[:2], len(times), fire.shape)
z=np.load('../runs/v4_noanchor/test_land_preds.npz')
T,ij,y,c,p=z["T_h1"],z["ij_h1"],z["y_h1"],z["c_h1"],z["p_h1"]
print("T dates", times[T.min()], times[T.max()])
slot=np.array([ (k % 46) for k in range(len(times))])
def clim(years):
    tr=[k for k,d in enumerate(times) if f"{years[0]}-01-01"<=d<=f"{years[1]}-12-31"]
    out={}
    for s in np.unique(slot[T]):
        ks=[k for k in tr if slot[k]==s]
        out[s]=fire[ks].mean(0).astype(np.float32)
    return out
tiles=lambda a,k: a[(k//100)*TILE:(k//100+1)*TILE,(k%100)*TILE:(k%100+1)*TILE].ravel()
land=np.stack([tiles(lsm,k)>0.5 for k in ij])
for yrs in [(2003,2017),(2002,2017)]:
    C=clim(yrs)
    cc=np.stack([tiles(C[slot[t]],k) for t,k in zip(T,ij)])
    print(yrs,"unsmoothed clim AUPRC land-only %.4f | all pixels in tiles %.4f"%(AP(y[land],cc[land]),AP(y.ravel(),cc.ravel())))
print("ours recency clim land %.4f all %.4f"%(AP(y[land],c[land]),AP(y.ravel(),c.ravel())))
print("ours model land %.4f all %.4f"%(AP(y[land],p[land]),AP(y.ravel(),p.ravel())))
# 평활(±w 슬롯, 순환) 균일 가중, 2002-2017
tr=[k for k,d in enumerate(times) if "2002-01-01"<=d<="2017-12-31"]
S={s: fire[[k for k in tr if slot[k]==s]].mean(0).astype(np.float32) for s in range(46)}
for w in (1,2,3):
    Cs={s: np.mean([S[(s+o)%46] for o in range(-w,w+1)],0) for s in np.unique(slot[T])}
    cc=np.stack([tiles(Cs[slot[t]],k) for t,k in zip(T,ij)])
    print("uniform 2002-2017 smoothed ±%d slots: land AUPRC %.4f"%(w,AP(y[land],cc[land])))
