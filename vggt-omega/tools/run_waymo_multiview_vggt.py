#!/usr/bin/env python3
"""Run official VGGT-Omega geometry inference on Waymo multi-camera clips."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import cv2, imageio.v2 as imageio, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo_gradio import load_model, run_model
from run_driving_demo import write_ply, voxel_filter, trajectory_png

def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--scene_dir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/vggt_omega_1b_512.pt')
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--views',type=int,choices=[3,5],required=True)
    p.add_argument('--timestamps',type=int,default=20); p.add_argument('--image_resolution',type=int,default=512)
    p.add_argument('--confidence_threshold',type=float,default=1.2); p.add_argument('--voxel_size',type=float,default=.05)
    p.add_argument('--pixel_stride',type=int,default=2); p.add_argument('--fps',type=float,default=10); p.add_argument('--max_video_points',type=int,default=18000)
    return p.parse_args()

def ordered_images(scene:Path, views:int, n:int, temp:Path):
    src=scene/'images_4' if (scene/'images_4').is_dir() else scene/'images'
    cams=list(range(5)) if views==5 else [1,0,2]
    out=temp/'images'; out.mkdir(parents=True,exist_ok=True); result=[]
    for t in range(n):
        for c in cams:
            cand=sorted(src.glob(f'{t:03d}_{c}.*'))
            if not cand: raise FileNotFoundError(f'missing timestamp={t}, camera={c} in {src}')
            dst=out/f'{t:06d}_cam{c}{cand[0].suffix.lower()}'; dst.write_bytes(cand[0].read_bytes()); result.append(dst)
    return result,cams

def color_depth(depth, valid):
    d=depth; vals=d[valid]; lo,hi=np.percentile(vals,(2,98)); cmap=plt.get_cmap('turbo')
    return cmap(np.clip((d-lo)/(hi-lo+1e-8),0,1))[...,:3],(float(lo),float(hi))

def main():
    a=parse(); a.output_dir.mkdir(parents=True,exist_ok=True); temp=a.output_dir/'_input'; paths,cams=ordered_images(a.scene_dir,a.views,a.timestamps,temp)
    model=load_model(str(a.checkpoint)); start=time.perf_counter()
    pred=run_model(str(temp),model,a.image_resolution); elapsed=time.perf_counter()-start
    np.savez_compressed(a.output_dir/'predictions.npz',**{k:v for k,v in pred.items() if isinstance(v,np.ndarray)})
    d=pred['depth'][...,0]; conf=pred['depth_conf']; valid=np.isfinite(d)&(d>0)&np.isfinite(conf)
    dvis,drange=color_depth(d,valid); (a.output_dir/'depth').mkdir(exist_ok=True)
    for i,x in enumerate(dvis): imageio.imwrite(a.output_dir/'depth'/f'{i:06d}.png',(x*255).astype('uint8'))
    writer=imageio.get_writer(a.output_dir/'depth.mp4',fps=a.fps,codec='libx264',quality=7,macro_block_size=1)
    for x in dvis: writer.append_data((x*255).astype('uint8'))
    writer.close()
    world=pred['world_points_from_depth']; images=pred['images']; images=np.transpose(images,(0,2,3,1)) if images.ndim==4 else images
    pts=[]; cols=[]; cs=[]; fs=[]
    for i in range(len(paths)):
        sl=(slice(None,None,a.pixel_stride),slice(None,None,a.pixel_stride)); m=valid[i][sl]&(conf[i][sl]>=0)
        pts.append(world[i][sl][m]); cols.append((images[i][sl][m].clip(0,1)*255).astype('uint8')); cs.append(conf[i][sl][m]); fs.append(np.full(m.sum(),i,np.int32))
    pts,cols,cs,fs=[np.concatenate(x) for x in (pts,cols,cs,fs)]; write_ply(a.output_dir/'scene_full.ply',pts,cols,cs,fs)
    keep=cs>=a.confidence_threshold; fpts,fcols,fcs,ffs=voxel_filter(pts[keep],cols[keep],cs[keep],fs[keep],a.voxel_size); write_ply(a.output_dir/'scene_filtered.ply',fpts,fcols,fcs,ffs)
    trajectory_png(pred['extrinsic'],a.output_dir/'camera_trajectory.png')
    # GLB point cloud for quick browser viewing.
    import trimesh
    cloud=trimesh.points.PointCloud(fpts,colors=fcols); cloud.export(a.output_dir/'scene.glb')
    rgb=[cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB) for p in paths]
    vh,vw=rgb[0].shape[:2]; writer=imageio.get_writer(a.output_dir/'scene_demo.mp4',fps=a.fps,codec='libx264',quality=7,macro_block_size=1)
    rng=np.random.default_rng(0); q=rng.choice(len(fpts),min(a.max_video_points,len(fpts)),replace=False); pp=fpts[q]
    for t in range(a.timestamps):
        fig=plt.figure(figsize=(16,9),dpi=120); gs=fig.add_gridspec(3,1,height_ratios=[1,1,1.2])
        for row,title,arr in [(0,'Input RGB',rgb),(1,'Predicted depth',dvis)]:
            ax=fig.add_subplot(gs[row,0]); tiles=[]
            for c in range(a.views):
                j=t*a.views+c; tile=arr[j]; tile=cv2.resize(tile,(vw//a.views,vh)) if a.views>1 else tile; tiles.append(tile)
            ax.imshow(np.concatenate(tiles,axis=1)); ax.axis('off'); ax.set_title(f'{title} | timestamp {t}')
        ax=fig.add_subplot(gs[2,0],projection='3d'); ax.scatter(pp[:,0],pp[:,1],pp[:,2],s=.2,c='#4c78a8',alpha=.3); ax.view_init(18,35+t*.8); ax.set_title('Fused world point cloud'); ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
        fig.tight_layout(); fig.canvas.draw(); writer.append_data(np.asarray(fig.canvas.buffer_rgba())[...,:3]); plt.close(fig)
    writer.close()
    meta={'scene':a.scene_dir.name,'cameras':cams,'camera_order':'timestamp-major','timestamps':a.timestamps,'num_images':len(paths),'checkpoint':str(a.checkpoint.resolve()),'image_resolution':list(pred['images'].shape[-2:]),'confidence_threshold':a.confidence_threshold,'voxel_size':a.voxel_size,'points_full':len(pts),'points_filtered':len(fpts),'elapsed_seconds':elapsed,'gpu_peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else None,'protocol':'official VGGT-Omega geometry inference; no RGB renderer'}
    (a.output_dir/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
