from pathlib import Path
import sys,pickle
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from PIL import Image
R=Path(__file__).resolve().parents[2];P=R/'experiments';V=P/'primary'
F=R/'outputs/figures';E=R/'outputs/attributions';F.mkdir(parents=True,exist_ok=True);E.mkdir(parents=True,exist_ok=True)
crop=R/'outputs/structure_crops';crop.mkdir(parents=True,exist_ok=True)
for a in ['HIC','Tm2','AC-SINS','PR_CHO','Titer','SMAC','HAC','Tm1','Tonset','SEC']:
 path=P/'cartoons/structures'/('residue_'+a+'.png')
 if not path.exists():raise FileNotFoundError('Render the cartoons with ChimeraX first: '+str(path))
 image=Image.open(path);box=image.getchannel('A').getbbox();image.crop(box).save(crop/path.name)
sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import aw_moe as AW
with open(V/'evidence/fitted_primary.pkl','rb') as f:z=pickle.load(f)
cols=z['descriptor_columns']+z['bio_columns'];g=z['g3']
plt.rcParams.update({'font.family':'Arial','font.size':6,'axes.titlesize':7,'axes.labelsize':6,'xtick.labelsize':5.5,'ytick.labelsize':5.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.5,'pdf.fonttype':42,'svg.fonttype':'none','xtick.major.size':2,'ytick.major.size':2})
COMP=['HIC','Tm2','AC-SINS','PR_CHO','Titer']; EXTRA=['SMAC','HAC','Tm1','Tonset','SEC']
nice=lambda a: {'PR_CHO':'PR-CHO'}.get(a,a)
def short(s):
    s=s.replace('MOE_properties::','MOE ').replace('TAP::','TAP ').replace('Aggrescan3D::','A3D ').replace('AntiFold::','AF ')
    for x,y in [('frac_aromatic','arom.'),('frac_aliphatic','aliph.'),('frac_polar','polar'),('frac_pos','pos.'),('frac_neg','neg.'),('deamidation_NG_NS','NG/NS'),('isomerization_DG_DS','DG/DS'),('hmoment','μH'),('aromatic_sasa','arom.SASA'),('charge_pH','charge')]:s=s.replace(x,y)
    return s.replace('_',' ').replace('aggrescan 90 score','score 90').replace('hyd strength cdr','CDR hyd.')

fig=plt.figure(figsize=(183/25.4,169/25.4))
fig.text(.015,.982,'A  AbDev-X named-feature attributions',fontsize=7,fontweight='bold',va='top')
fig.text(.015,.958,'Actual four-expert predictor; training-mean reference. Feature value: low (blue) to high (red).',fontsize=6,va='top')
tables=[]
for j,a in enumerate(COMP):
    m=z['models'][a];w=np.r_[m['weights']['desc'],m['weights']['bio']]
    x=np.hstack([z['B3']['desc'],z['B3']['bio']]);train=np.hstack([z['B1']['desc'],z['B1']['bio']])
    ycol=AW.GENUINE_ASSAYS.get(a,(a,''))[0];mu=train[np.isfinite(z['train_df'][ycol].to_numpy(float))].mean(0)
    phi=(x-mu)*w; selected=np.argsort(np.mean(abs(phi),axis=0))[-12:]
    ax=fig.add_axes([.105+j*.198,.585,.088,.325])
    for t,c in enumerate(selected):
        values=phi[:,c];span=max(np.ptp(values),1e-9);bins=np.floor((values-values.min())/span*15).astype(int);offset=np.zeros(len(values))
        for b in np.unique(bins):
            ids=np.flatnonzero(bins==b); levels=np.arange(len(ids))-(len(ids)-1)/2
            offset[ids]=levels*.035
        offset=np.clip(offset,-.30,.30)
        ax.scatter(values,t+offset,c=rankdata(x[:,c])/len(x),cmap='coolwarm',vmin=0,vmax=1,s=2.1,alpha=.72,linewidths=0)
        tables.append({'assay':a,'feature':cols[c],'label':short(cols[c]),'mean_abs_attribution':np.mean(abs(values))})
    ax.set_yticks(range(12),[short(cols[c]) for c in selected]);ax.tick_params(axis='y',length=0,pad=2)
    ax.axvline(0,c='#aaaaaa',lw=.5,zorder=-1);ax.set_ylim(-.6,11.6);ax.set_title(nice(a),fontweight='bold',pad=5);ax.set_xlabel('Contribution',labelpad=2)
    ax.locator_params(axis='x',nbins=2)
pd.DataFrame(tables).to_csv(E/'figure_named_feature_labels.csv',index=False)

for label,assays,y in [('B',EXTRA,.305),('C',COMP,.065)]:
    fig.text(.015,y+.238,f'{label}  AbDev-X residue contributions through sequence descriptors',fontsize=7,fontweight='bold',va='top')
    for j,a in enumerate(assays):
        ax=fig.add_axes([.015+j*.198,y,.19,.205]);ax.imshow(plt.imread(crop/f'residue_{a}.png'));ax.axis('off');ax.set_title(nice(a),fontsize=7,fontweight='bold',pad=0)
cb=fig.add_axes([.36,.039,.26,.009]);fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-1,1),cmap='bwr'),cax=cb,orientation='horizontal',ticks=[-1,0,1]);cb.tick_params(labelsize=5.5,pad=1)
fig.text(.015,.009,'Blue: lowers predicted assay value; red: raises it. Residue scores scaled within assay; PLM, bio and subtype fixed.',fontsize=6)
for ext in ['pdf','png','svg']:fig.savefig(F/f'main06_primary_attribution.{ext}',dpi=350)
plt.close(fig)

