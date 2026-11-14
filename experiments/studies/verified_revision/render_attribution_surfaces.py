"""Render real PDB solvent-accessible envelopes with scipy and matplotlib.
No commercial renderer is required. Surface is the union of atomic van der Waals
spheres inflated by a 1.4 A probe, meshed using marching tetrahedra at 0.8 A spacing.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
H=Path(__file__).parent;E=H/'evidence';F=H/'figures';meta=json.loads((E/'residue_shap_manifest.json').read_text());d=pd.read_csv(E/'primary_residue_shap.csv')
pdb=PDBParser(QUIET=True).get_structure('ab',meta['structure']);atoms=[a for a in pdb.get_atoms() if a.element!='H'];xyz=np.array([a.coord for a in atoms]);rad=np.array([{'C':1.70,'N':1.55,'O':1.52,'S':1.8}.get(a.element,1.7)+1.4 for a in atoms]);keys=[(a.parent.parent.id,str(a.parent.id[1])+a.parent.id[2].strip()) for a in atoms]
center=xyz.mean(0);_,_,vt=np.linalg.svd(xyz-center,full_matrices=False);xyz=(xyz-center)@vt.T;tree=cKDTree(xyz);spacing=.8
lo=xyz.min(0)-4;hi=xyz.max(0)+4;grids=[np.arange(lo[i],hi[i]+spacing,spacing) for i in range(3)];shape=tuple(map(len,grids));points=np.stack(np.meshgrid(*grids,indexing='ij'),axis=-1);dist,idx=tree.query(points.reshape(-1,3),k=8);field=(dist-rad[idx]).min(1).reshape(shape)
corners=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]])
base=np.stack(np.meshgrid(*[np.arange(n-1) for n in shape],indexing='ij'),axis=-1).reshape(-1,3);ii=base[:,None,:]+corners;vals=field[ii[:,:,0],ii[:,:,1],ii[:,:,2]];keep=(vals.min(1)<=0)&(vals.max(1)>=0);base=base[keep];vals=vals[keep];coords=lo+(base[:,None,:]+corners)*spacing
tetra=[[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]];triangles=[]
for tet in tetra:
 v=vals[:,tet];c=coords[:,tet];case=(v<0)@np.array([1,2,4,8])
 for mask in range(1,15):
  sel=case==mask
  if not sel.any():continue
  vv=v[sel];cc=c[sel];ins=[i for i in range(4) if mask&(1<<i)];outs=[i for i in range(4) if not mask&(1<<i)]
  def edge(i,j):
   t=vv[:,i]/(vv[:,i]-vv[:,j]);return cc[:,i]+t[:,None]*(cc[:,j]-cc[:,i])
  if len(ins)==1:triangles.append(np.stack([edge(ins[0],j) for j in outs],axis=1))
  elif len(outs)==1:triangles.append(np.stack([edge(outs[0],j) for j in ins],axis=1))
  else:
   a,b=ins;c0,c1=outs;p00=edge(a,c0);p01=edge(a,c1);p10=edge(b,c0);p11=edge(b,c1);triangles.extend([np.stack([p00,p01,p10],axis=1),np.stack([p01,p11,p10],axis=1)])
tri=np.concatenate(triangles);near=tree.query(tri.mean(1))[1];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm/=np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1e-10);light=np.array([.5,-.4,.75]);light/=np.linalg.norm(light);shade=.65+.35*np.abs(norm@light)
def render(colors,file):
 fig=plt.figure(figsize=(3.2,2.7),dpi=600);ax=fig.add_axes([0,0,1,1],projection='3d');colors=colors.copy();colors[:,:3]*=shade[:,None];poly=Poly3DCollection(tri,facecolors=colors,edgecolors='none',linewidths=0,antialiased=False);ax.add_collection3d(poly);bound=np.max(np.ptp(xyz,axis=0))/2+3;ax.set(xlim=(-bound,bound),ylim=(-bound,bound),zlim=(-bound,bound));ax.set_box_aspect([1,1,1]);ax.view_init(elev=20,azim=-55);ax.set_proj_type('ortho');ax.axis('off');fig.savefig(file,transparent=True,dpi=600);plt.close(fig);im=Image.open(file);im.crop(im.getchannel('A').getbbox()).save(file)
for a,z in d.groupby('assay',sort=False):
 scores={(r.chain,str(r.pdb_residue)):r.shap for _,r in z.iterrows()};s=np.array([scores[k] for k in keys]);scale=max(np.quantile(abs(z.shap),.95),1e-12);colors=plt.cm.RdBu_r((np.clip(s[near]/scale,-1,1)+1)/2);render(colors,F/f'residue_{a}.png');print('Rendered',a,flush=True)
colors=np.array([matplotlib.colors.to_rgba('#0072B2' if keys[i][0]=='H' else '#009E73') for i in near]);render(colors,F/'representative_fv.png')
(E/'surface_render_manifest.json').write_text(json.dumps({'method':'union of atom van der Waals spheres inflated by 1.4 Angstrom probe; marching tetrahedra','grid_spacing_A':spacing,'triangles':len(tri),'color_assignment':'score of residue belonging to the nearest atom at each triangle centroid','structure':meta['structure']},indent=2))
