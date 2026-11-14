from pathlib import Path
import sys,pickle,re,shutil,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from PIL import Image
P=Path(__file__).resolve().parent;OLD=P/'bmc_formatted_revision';O=P/'layout_chimerax_revision';F=O/'figures';E=O/'evidence';V=P/'verified_revision'
E.mkdir(parents=True,exist_ok=True);shutil.copytree(OLD/'figures',F,dirs_exist_ok=True)
# Crop only empty transparent borders from the newly generated ChimeraX renders.
crop=O/'structures/cropped';crop.mkdir(exist_ok=True)
for image in (O/'structures').glob('residue_*.png'):
 im=Image.open(image);box=im.getchannel('A').getbbox();assert box
 im.crop((max(0,box[0]-15),max(0,box[1]-15),min(im.width,box[2]+15),min(im.height,box[3]+15))).save(crop/image.name)
sys.path[:0]=[str(P.parents[1]/'scripts'),str(P.parents[1]/'src')]
import aw_moe as AW
with open(V/'evidence/fitted_primary.pkl','rb') as file:z=pickle.load(file)
cols=z['descriptor_columns']+z['bio_columns'];g=z['g3']
source=(P/'update_model_structure_attributions.py').read_text(encoding='utf8')
part=source[source.index('plt.rcParams.update'):source.index("s=(OLD/'AbDevX_evidence_led_narrative_main_SI.tex')")]
part=part.replace("V/'figures'/f'residue_{a}.png'","crop/f'residue_{a}.png'").replace("cmap='RdBu_r'","cmap='bwr'")
exec(part,globals())
text=(OLD/'AbDevX_BMC_Bioinformatics_main_SI.tex').read_text(encoding='utf8')
architecture=re.search(r'^\\figpanel\{fig01_architecture_user\.pdf\}.*$',text,re.M).group(0)
text=text.replace(architecture+'\n','',1)
text=text.replace(r'\FloatBarrier','')
text=text.replace(r'\thispagestyle{empty}\clearpage','')
text=text.replace(r'\end{abstract}'+'\n'+r'\textbf{Keywords:',r'\end{abstract}'+'\n'+r'\textbf{Keywords:')
# Keep the main/SI boundary but remove all other hard page breaks.
text=text.replace(r'\clearpage\appendix','KEEP_MAIN_SI_BOUNDARY')
text=text.replace(r'\clearpage','')
text=text.replace('KEEP_MAIN_SI_BOUNDARY',r'\clearpage\appendix')
anchor=r'\section{Methods}'
intro_figure=r'Figure~\ref{fig:architecture} summarizes the sequence-to-representation workflow, the four expert architectures, and the final assay-specific predictions.'+'\n\n'+architecture.replace(r'\figpanel',r'\archpanel',1)+'\n\n'
text=text.replace('Recent evidence identifies two obstacles.',intro_figure+'Recent evidence identifies two obstacles.',1)
text=text.replace(r'\newcommand{\AbDev}',r'\newcommand{\archpanel}[4]{\begin{figure}[H]\centering\includegraphics[width=\textwidth]{#1}\caption{#2}\label{#3}\end{figure}#4}'+'\n'+r'\newcommand{\AbDev}',1)
text=text.replace('{fig:structure_predictors}\n','{fig:structure_predictors}'+r'\FloatBarrier'+'\n')
text=text.replace(r'\small'+'\n'+r'\setlength{\tabcolsep}{5pt}'+'\n'+r'\begin{longtable}',r'\clearpage\small'+'\n'+r'\setlength{\tabcolsep}{5pt}'+'\n'+r'\begin{longtable}',1)
text=text.replace(r'\end{longtable}'+'\n'+r'\begin{longtable}',r'\end{longtable}'+'\n'+r'\clearpage\begin{longtable}',1)
text=text.replace(r'\section{Discussion}',r'\FloatBarrier'+'\n'+r'\section{Discussion}',1)
text=text.replace(r'\section*{Declarations}',r'\FloatBarrier'+'\n'+r'\section*{Declarations}',1)
text=text.replace(r'\setcounter{topnumber}{2}',r'''\raggedbottom
\setcounter{topnumber}{4}
\setcounter{bottomnumber}{3}
\setcounter{totalnumber}{5}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\floatpagefraction}{0.65}
\setlength{\textfloatsep}{12pt plus 2pt minus 3pt}
\setlength{\floatsep}{10pt plus 2pt minus 2pt}
\makeatletter
\setlength{\@fptop}{0pt}
\setlength{\@fpbot}{0pt plus 1fil}
\makeatother''')
text=text.replace(r'\renewcommand{\topfraction}{0.9}',r'\renewcommand{\topfraction}{0.98}').replace(r'\renewcommand{\textfraction}{0.07}',r'\renewcommand{\textfraction}{0.02}')
text=text.replace('rendered as PyMOL cartoons','rendered as UCSF ChimeraX cartoons')
old='Cartoons were ray-traced using PyMOL through pymol2 after secondary-structure assignment with DSS. Beta strands, helices, and loops are shown with residue-level attribution colors. Residue scores were assigned to the B-factor field, colored by residue using a blue--white--red palette, and saved as PDB and PyMOL session files.'
new=r'Cartoons were rendered using UCSF ChimeraX 1.11.1 after secondary-structure assignment with DSSP~\citep{chimerax}. Beta strands, helices, and loops are shown with residue-level attribution colors. The existing signed residue scores were mapped directly to residue cartoon colors using a blue--white--red palette; rendered images and ChimeraX session files retain the same underlying score table.'
assert old in text;text=text.replace(old,new)
text=text.replace(r'\end{thebibliography}',r'''\bibitem{chimerax} Meng EC, Goddard TD, Pettersen EF, Couch GS, Pearson ZJ, Morris JH, et al. UCSF ChimeraX: Tools for structure building and analysis. Protein Sci. 2023;32:e4792. \url{https://doi.org/10.1002/pro.4792}.
\end{thebibliography}''')
# Reorder Vancouver bibliography after moving the architecture and adding a tool.
begin=text.index(r'\begin{thebibliography}{99}');end=text.index(r'\end{thebibliography}',begin)+len(r'\end{thebibliography}')
entries={m.group(1):m.group(0).strip() for m in re.finditer(r'\\bibitem\{([^}]+)\}.*?(?=\\bibitem|\\end\{thebibliography\})',text[begin:end],re.S)}
order=[]
for m in re.finditer(r'\\cite(?:p|t)?(?:\[[^\]]*\])*\{([^}]+)\}',text[:begin]+text[end:]):
 for key in m.group(1).split(','):
  if key not in order:order.append(key)
order += [key for key in entries if key not in order]
text=text[:begin]+r'\begin{thebibliography}{99}'+'\n'+'\n'.join(entries[k] for k in order)+'\n'+r'\end{thebibliography}'+text[end:]
assert text.index(r'\archpanel{fig01_architecture_user.pdf}')<text.index(r'\section{Methods}')
assert text.count(r'\archpanel{fig01_architecture_user.pdf}')==1
(O/'AbDevX_BMC_clean_layout_main_SI.tex').write_text(text,encoding='utf8')
print('Ready: architecture in Background; ten ChimeraX cartoons; unnecessary hard page breaks and float barriers removed')
