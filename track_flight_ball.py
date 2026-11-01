import cv2, csv, argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument('video',nargs='?',default='flight_camera.avi')
ap.add_argument('--out',default='flight_tracking')
args=ap.parse_args()
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
cap=cv2.VideoCapture(args.video)
if not cap.isOpened():raise RuntimeError('動画を開けません: '+args.video)
fps=cap.get(cv2.CAP_PROP_FPS) or 25
W=int(cap.get(3));H=int(cap.get(4))
writer=cv2.VideoWriter(str(out/'tracked.avi'),cv2.VideoWriter_fourcc(*'MJPG'),fps,(W,H))
if not writer.isOpened():raise RuntimeError('動画保存を開始できません')
rows=[];prev=None;lost=0;frames=[]
try:
 idx=0
 while True:
  ok,im=cap.read()
  if not ok:break
  t=idx/fps
  g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  # 明るい背景にある暗色領域を連結成分で抽出
  m=cv2.inRange(g,0,85)
  n,lab,stats,cent=cv2.connectedComponentsWithStats(m)
  cand=[]
  for k in range(1,n):
   x,y,w,h,area=map(int,stats[k]); cx,cy=map(float,cent[k]); ratio=w/h
   if not(90<=area<=1500 and 10<=w<=50 and 10<=h<=50 and .48<=ratio<=1.65):continue
   if x<275 or x+w>W-7 or y<7 or y+h>H-7:continue
   # 背景の白さを確認: 対象の周囲リング
   pad=6; xa=max(0,x-pad); xb=min(W,x+w+pad);ya=max(0,y-pad);yb=min(H,y+h+pad)
   ring=g[ya:yb,xa:xb].copy(); ring[y-ya:y-ya+h,x-xa:x-xa+w]=0
   sel=np.ones(ring.shape,dtype=bool);sel[y-ya:y-ya+h,x-xa:x-xa+w]=False
   if np.mean(ring[sel]>105)<.55:continue
   cand.append((cx,cy,area,w,h))
  choice=None
  # 撮影開始直後は開始位置から探索、途切れたときも再捕捉
  if prev is None:
   # 初期は吊り下げ位置。飛翔中は探索結果が一意の場合のみ再捕捉
   ref=(325,410) if t<1.4 else None
   if ref is not None:
    cs=[v for v in cand if np.hypot(v[0]-ref[0],v[1]-ref[1])<65]
    if cs:choice=min(cs,key=lambda v:np.hypot(v[0]-ref[0],v[1]-ref[1]))
   elif len(cand)==1:choice=cand[0]
  else:
   # 最大速度: 約50px / 40ms。再捕捉は慎重に行う
   limit=64 if lost<2 else 85
   near=[v for v in cand if np.hypot(v[0]-prev[0],v[1]-prev[1])<limit]
   if near:choice=min(near,key=lambda v:np.hypot(v[0]-prev[0],v[1]-prev[1]))
  if choice:
   cx,cy,area,w,h=choice;prev=(cx,cy);lost=0;status='detected'
   cv2.circle(im,(round(cx),round(cy)),max(w,h)//2+4,(0,255,0),2)
  else:
   cx=cy=area=None;lost+=1;status='missing'
   if lost>=4:prev=None
  rows.append((idx,round(t,4),'' if cx is None else round(cx,2),'' if cy is None else round(cy,2),'' if area is None else area,status))
  cv2.putText(im,f'frame={idx} t={t:.2f}s {status}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
  writer.write(im)
  if idx%4==0 and .8<=t<=2.8:
   small=cv2.resize(im,(320,240));frames.append((idx,small))
  idx+=1
finally:
 cap.release();writer.release()
with (out/'ball_positions.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['frame','time_s','x_px','y_px','area_px','status']);w.writerows(rows)
xx=[(r[1],r[2],r[3]) for r in rows if r[5]=='detected' and .8<=r[1]<=2.8]
if xx:
 a=np.array(xx,float);plt.figure(figsize=(9,4.5));plt.plot(a[:,0],a[:,1],'.-',label='X');plt.plot(a[:,0],a[:,2],'.-',label='Y');plt.xlabel('Time from recording start (s)');plt.ylabel('Image coordinates (px)');plt.legend();plt.grid();plt.tight_layout();plt.savefig(out/'ball_trajectory.png',dpi=150);plt.close()
if frames:
 cols=4; rowsN=(len(frames)+cols-1)//cols
 sheet=np.full((rowsN*270,cols*320,3),255,np.uint8)
 for j,(ind,frm) in enumerate(frames):
  y=(j//cols)*270;x=(j%cols)*320
  sheet[y:y+240,x:x+320]=frm
  cv2.putText(sheet,f'{ind/fps:.2f}s',(x+8,y+258),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
 cv2.imwrite(str(out/'contact_sheet.jpg'),sheet)
print('動画フレーム:',len(rows),'全体検出数:',sum(r[5]=='detected' for r in rows))
print('飛翔区間 1.4-2.3s:',sum(1 for r in rows if 1.4<=r[1]<=2.3 and r[5]=='detected'), '/',sum(1 for r in rows if 1.4<=r[1]<=2.3))
print('保存先:',out)
