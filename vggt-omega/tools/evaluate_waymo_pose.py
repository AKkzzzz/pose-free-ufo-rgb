#!/usr/bin/env python3
"""Evaluate VGGT predicted camera centers against the local Waymo pose labels."""
import argparse,json
from pathlib import Path
import numpy as np

def umeyama(x,y):
    # x -> y similarity transform
    mx,my=x.mean(0),y.mean(0); X=x-mx; Y=y-my
    U,S,Vt=np.linalg.svd(X.T@Y/len(x)); D=np.eye(3); D[-1,-1]=np.sign(np.linalg.det(U@Vt)); R=U@D@Vt
    scale=(S*D.diagonal()).sum()/(X**2).sum(); t=my-scale*(R@mx); return scale,R,t

def main():
    p=argparse.ArgumentParser(); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--annotation',type=Path,required=True); p.add_argument('--views',type=int,required=True); p.add_argument('--timestamps',type=int,required=True); p.add_argument('--cameras',nargs='+',required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    pred=np.load(a.predictions); ext=pred['extrinsic']; centers=[]
    ann=json.loads(a.annotation.read_text()); gt=[]
    for t in range(a.timestamps):
        for c in a.cameras:
            M=np.asarray(ann['camera_to_world'][str(c)][t],dtype=np.float64); gt.append(M[:3,3])
    for E in ext:
        R=E[:3,:3]; tt=E[:3,3]; centers.append(-R.T@tt)
    x=np.asarray(centers); y=np.asarray(gt); s,R,t=umeyama(x,y); xa=(s*(x@R.T))+t; err=np.linalg.norm(xa-y,axis=1)
    # Relative translation errors are scale-normalized by the GT step length.
    rel=[]
    for i in range(len(x)-a.views):
        dp=xa[i+a.views]-xa[i]; dg=y[i+a.views]-y[i]; rel.append(np.linalg.norm(dp-dg)/(np.linalg.norm(dg)+1e-6))
    # Same-timestamp camera baseline: compare pairwise distances, invariant to global Sim3.
    base=[]
    for t0 in range(a.timestamps):
        ids=range(t0*a.views,(t0+1)*a.views)
        for i in ids:
            for j in ids:
                if j>i: base.append(abs(np.linalg.norm(xa[i]-xa[j])-np.linalg.norm(y[i]-y[j])))
    out={'num_cameras':len(x),'sim3_scale':float(s),'ate_rmse_m':float(np.sqrt(np.mean(err**2))),'ate_median_m':float(np.median(err)),'ate_p90_m':float(np.percentile(err,90)),'relative_translation_error_mean':float(np.mean(rel)),'relative_translation_error_median':float(np.median(rel)),'rig_baseline_abs_error_mean_m':float(np.mean(base)),'rig_baseline_abs_error_max_m':float(np.max(base)),'depth_lidar_metrics':'unavailable: local STORM Waymo subset contains RGB, masks, and pose labels but no LiDAR/depth ground truth'}
    a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
