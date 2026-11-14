"""Editable multi-panel architecture: overview plus four operational experts."""
from pathlib import Path
from io import BytesIO
import math,re
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
H=Path(__file__).parent;OUT=H/'AbDevX_architecture_panels_v5.pptx'
P=Presentation();P.slide_width=Inches(16);P.slide_height=Inches(12)
NAVY='18354C';INK='243746';LINE='688094';MUTED='5C6D79';PALE='F3F7FA'
C=['C74D55','4F83B6','76A644','DA8B39'];BG=['FDEBED','EAF2FC','EFF7E5','FFF1E2']
assets={}
src=Presentation(H/'AbDevX_architecture_code_matched.pptx')
def pics(ss):
 for s in ss:
  if s.shape_type==6:pics(s.shapes)
  elif s.shape_type==13:assets[s.shape_id]=s.image.blob
pics(src.slides[0].shapes)
def color(s):return RGBColor.from_string(s)
def txt(sl,x,y,w,h,s,size=14,bold=False,c=INK,align=PP_ALIGN.LEFT,font='Arial'):
 sh=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.clear();tf.word_wrap=True
 tf.margin_left=tf.margin_right=Inches(.02);tf.margin_top=tf.margin_bottom=0;tf.vertical_anchor=MSO_ANCHOR.MIDDLE
 for i,line in enumerate(s.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.alignment=align;p.space_before=p.space_after=Pt(0)
  # Native PowerPoint rich-text baseline shifts keep all mathematical indices editable.
  pos=0
  for m in re.finditer(r'([_^])\{([^{}]+)\}',line):
   if m.start()>pos:
    r=p.add_run();r.text=line[pos:m.start()];r.font.name=font;r.font.size=Pt(size);r.font.bold=bold;r.font.color.rgb=color(c)
   r=p.add_run();r.text=m[2];r.font.name=font;r.font.size=Pt(size*.8);r.font.bold=bold;r.font.color.rgb=color(c);r._r.get_or_add_rPr().set('baseline','-25000' if m[1]=='_' else '35000');pos=m.end()
  if pos<len(line):
   r=p.add_run();r.text=line[pos:];r.font.name=font;r.font.size=Pt(size);r.font.bold=bold;r.font.color.rgb=color(c)
 return sh
def shape(sl,x,y,w,h,kind=MSO_SHAPE.ROUNDED_RECTANGLE,fill='FFFFFF',stroke=LINE,lw=.8):
 sh=sl.shapes.add_shape(kind,Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=color(fill);sh.line.color.rgb=color(stroke);sh.line.width=Pt(lw)
 if kind==MSO_SHAPE.ROUNDED_RECTANGLE:
  try:sh.adjustments[0]=.12
  except Exception:pass
 return sh
def box(sl,x,y,w,h,s,fill=PALE,stroke=LINE,size=14,kind=MSO_SHAPE.ROUNDED_RECTANGLE,bold=False):
 shape(sl,x,y,w,h,kind,fill,stroke);return txt(sl,x+.05,y+.025,w-.10,h-.05,s,size,bold,align=PP_ALIGN.CENTER)
def line(sl,x1,y1,x2,y2,arrow=True,c=LINE,dash=False,lw=1):
 sh=sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2));sh.line.color.rgb=color(c);sh.line.width=Pt(lw)
 ln=sh._element.spPr.get_or_add_ln()
 if arrow:
  end=OxmlElement('a:tailEnd');end.set('type','triangle');end.set('w','sm');end.set('len','sm');ln.append(end)
 if dash:
  d=OxmlElement('a:prstDash');d.set('val','dash');ln.append(d)
 return sh
class Panel:
 def __init__(self,sl,x,y,w,h,letter,title,accent=NAVY):
  self.sl=sl;self.x=x;self.y=y;self.w=w;self.h=h;self.k=w/7.4
  shape(sl,x,y,w,h,fill='FFFFFF',stroke='CCD8E0',lw=.8)
  shape(sl,x+.12*self.k,y+.13*self.k,.36*self.k,.36*self.k,MSO_SHAPE.OVAL,accent,accent)
  txt(sl,x+.12*self.k,y+.13*self.k,.36*self.k,.36*self.k,letter,15*self.k,True,'FFFFFF',PP_ALIGN.CENTER)
  txt(sl,x+.61*self.k,y+.10*self.k,w-.8*self.k,.43*self.k,title,17*self.k,True,NAVY)
 def t(self,x,y,w,h,s,size=14,bold=False,c=INK,align=PP_ALIGN.LEFT,font='Arial'):return txt(self.sl,self.x+x*self.k,self.y+y*self.k,w*self.k,h*self.k,s,size*self.k,bold,c,align,font)
 def b(self,x,y,w,h,s,fill=PALE,stroke=LINE,size=14,kind=MSO_SHAPE.ROUNDED_RECTANGLE,bold=False):return box(self.sl,self.x+x*self.k,self.y+y*self.k,w*self.k,h*self.k,s,fill,stroke,size*self.k,kind,bold)
 def a(self,x,y,xx,yy,**kw):return line(self.sl,self.x+x*self.k,self.y+y*self.k,self.x+xx*self.k,self.y+yy*self.k,**kw)
 def matrix(self,x,y,w,h,rows=4,cols=6,selected=None):
  gap=.035;cw=(w-gap*(cols-1))/cols;ch=(h-gap*(rows-1))/rows
  for r in range(rows):
   for c in range(cols):shape(self.sl,self.x+(x+c*(cw+gap))*self.k,self.y+(y+r*(ch+gap))*self.k,cw*self.k,ch*self.k,MSO_SHAPE.RECTANGLE,('81A9C5' if selected is None or c in selected else 'E6EDF2'),'FFFFFF',.2)
def new(title,subtitle=''):
 sl=P.slides.add_slide(P.slide_layouts[6]);sl.background.fill.solid();sl.background.fill.fore_color.rgb=color('FFFFFF')
 txt(sl,.35,.11,15.3,.43,title,24,True,NAVY)
 if subtitle:txt(sl,.36,.58,15.2,.34,subtitle,14,c=MUTED)
 return sl
def e1(sl,x,y,w,h):
 q=Panel(sl,x,y,w,h,'B','E1 | Best-source selector',C[0]);names=['ESM-2','AbLang2','Descriptors','HC/LC subtype','Merged bio']
 q.t(.20,.59,2.18,.30,'Five operational sources',13,True)
 q.t(2.62,.59,1.00,.30,'Ridge',13,True,align=PP_ALIGN.CENTER);q.t(3.94,.59,.82,.30,'CV ρ',13,True,align=PP_ALIGN.CENTER)
 for i,n in enumerate(names):
  yy=.98+i*.355;q.b(.22,yy,1.98,.29,n,BG[0],C[0],13);q.a(2.22,yy+.145,2.63,yy+.145);q.b(2.65,yy,.92,.29,'Head '+str(i+1),size=13);q.a(3.59,yy+.145,3.97,yy+.145);q.b(3.99,yy,.66,.29,'ρ_{'+str(i+1)+'}',fill='FFFFFF',stroke=C[0],size=13);q.a(4.65,yy+.145,4.75,yy+.145,arrow=False)
 q.a(4.75,1.12,4.75,2.55,arrow=False);q.a(4.65,1.12,4.75,1.12,arrow=False);q.a(4.65,2.55,4.75,2.55,arrow=False);q.a(4.75,1.83,5.05,1.83)
 q.b(5.05,1.36,1.05,.94,'Best\nCV ρ',BG[0],C[0],14,MSO_SHAPE.DIAMOND,True);q.a(6.1,1.83,6.34,1.83);q.b(6.36,1.28,.78,.98,'Refit\nbest\nhead',size=13)
 q.a(6.75,2.29,6.75,2.46,c=C[0]);q.b(6.28,2.48,.94,.35,'ŷ_{E1}',BG[0],C[0],17,MSO_SHAPE.OVAL,True)
 q.t(.22,2.91,6.95,.31,'Choose once per assay using inner OOF scores; then refit on GDPa1.',13,c=MUTED)
 return q
def mgfs(sl,x,y,w,h):
 q=Panel(sl,x,y,w,h,'C','A-MGFS | Assay-masked descriptors',C[1])
 q.t(.20,.66,1.25,.51,'Descriptors\n+ merged bio',13,True,align=PP_ALIGN.CENTER);q.matrix(.23,1.28,1.12,.80,4,7)
 q.a(1.43,1.68,1.73,1.68);q.b(1.75,1.19,1.07,.98,'Assay\nkeyword\nmask',BG[1],C[1],14,MSO_SHAPE.TRAPEZOID,True)
 q.t(1.72,.71,1.17,.31,'Mask rules',13,c=C[1],align=PP_ALIGN.CENTER);q.a(2.84,1.68,3.12,1.68);q.matrix(3.16,1.28,.70,.80,4,3);q.t(2.96,2.16,1.12,.48,'Retained features',13,align=PP_ALIGN.CENTER)
 q.a(3.91,1.68,4.05,1.68,arrow=False);q.a(4.05,1.68,4.05,1.25,arrow=False);q.a(4.05,1.25,4.18,1.25);q.b(4.22,1.04,1.25,.42,'Scale',BG[1],C[1]);q.a(4.85,1.48,4.85,1.66);q.b(4.22,1.69,1.25,.42,'PCA^{*}',BG[1],C[1]);q.a(4.85,2.13,4.85,2.31);q.b(4.22,2.34,1.25,.42,'RidgeCV',BG[1],C[1]);q.a(5.49,2.55,5.89,2.55);q.b(5.94,2.25,1.11,.58,'ŷ_{A}',BG[1],C[1],16,MSO_SHAPE.OVAL,True)
 q.t(.22,2.69,3.75,.49,'HIC: hydrophobic / aromatic terms\nAC-SINS: charge / pI terms',13,c=MUTED)
 return q
def dstack(sl,x,y,w,h):
 q=Panel(sl,x,y,w,h,'D','D-Stack | Two-stream ridge stacking',C[2])
 for yy,inputs,label in [(1.00,'ESM-2 + AbLang2','PLM head'),(2.00,'Descriptors + bio','Bio head')]:
  q.b(.23,yy,1.84,.62,inputs,BG[2],C[2],14);q.a(2.10,yy+.31,2.42,yy+.31);q.b(2.45,yy,1.11,.62,label,BG[2],C[2],14);q.a(3.59,yy+.31,4.02,yy+.31)
 q.t(3.90,.63,1.0,.30,'Inner OOF',13,True,align=PP_ALIGN.CENTER);q.matrix(4.07,1.10,.60,1.43,5,2);q.t(3.83,2.65,1.05,.30,'OOF pair',13,align=PP_ALIGN.CENTER)
 q.a(4.72,1.82,5.03,1.82);q.b(5.07,1.35,1.17,.94,'Ridge\nmeta-head\nα = 1',BG[2],C[2],14,MSO_SHAPE.HEXAGON,True);q.a(6.27,1.82,6.52,1.82);q.b(6.54,1.50,.66,.63,'ŷ_{D}',BG[2],C[2],17,MSO_SHAPE.OVAL,True)
 q.t(.23,2.98,6.92,.26,'Train meta-head on OOF predictions; refit both streams on GDPa1.',13,c=MUTED)
 return q
def fmoe(sl,x,y,w,h):
 q=Panel(sl,x,y,w,h,'E','F-MoE | Mechanistic group mixture',C[3])
 q.t(.23,.61,2.60,.31,'Example: HAC groups',13,True)
 groups=['Hydrophobic','Charge','Surface','Stability']
 for i,g in enumerate(groups):
  yy=1.24+i*.40;q.b(.24,yy,1.51,.32,g,BG[3],C[3],13);q.a(1.78,yy+.16,2.06,yy+.16);q.b(2.09,yy,1.04,.32,'Ridge',size=13);q.a(3.15,yy+.16,3.42,yy+.16);q.b(3.45,yy,.63,.32,'ŷ_{'+str(i+1)+'}',fill='FFFFFF',stroke=C[3],size=13);q.a(4.12,yy+.16,4.35,yy+.16);q.b(4.39,yy,.33,.32,'×',BG[3],C[3],16,MSO_SHAPE.OVAL);q.a(4.74,yy+.16,5.13,yy+.16,arrow=False)
  q.a(4.26,yy-.07,4.56,yy-.07,arrow=False,dash=True,c=C[3]);q.a(4.56,yy-.07,4.56,yy-.01,dash=True,c=C[3])
 q.b(2.95,.67,1.17,.36,'Group CV ρ',fill='FFFFFF',stroke=C[3],size=12.5);q.a(4.14,.85,4.41,.85);q.b(4.45,.63,1.65,.45,'Softmax · τ = 0.1',BG[3],C[3],13)
 q.a(4.45,1.0,4.26,1.0,arrow=False,dash=True,c=C[3]);q.a(4.26,1.0,4.26,2.37,arrow=False,dash=True,c=C[3]);q.t(6.10,1.10,1.10,.27,'weights w_{g}',12.5,c=C[3])
 q.a(5.15,1.40,5.15,2.60,arrow=False);q.a(5.15,1.96,5.48,1.96);q.b(5.51,1.61,.67,.67,'Σ',BG[3],C[3],24,MSO_SHAPE.OVAL,True);q.a(6.21,1.96,6.48,1.96);q.b(6.50,1.64,.67,.64,'ŷ_{F}',BG[3],C[3],17,MSO_SHAPE.OVAL,True)
 q.t(.23,2.98,6.92,.26,'Fixed assay-level group weights; eligibility changes with the assay.',13,c=MUTED)
 return q
def overview(sl,x,y,w,h):
 # Overview uses a wider logical canvas than the expert panels.
 shape(sl,x,y,w,h,fill='FFFFFF',stroke='CCD8E0');k=w/15.2
 def t(xx,yy,ww,hh,s,size=14,bold=False,c=INK,align=PP_ALIGN.LEFT,font='Arial'):return txt(sl,x+xx*k,y+yy*k,ww*k,hh*k,s,size*k,bold,c,align,font)
 def b(xx,yy,ww,hh,s,fill=PALE,stroke=LINE,size=14,kind=MSO_SHAPE.ROUNDED_RECTANGLE,bold=False):return box(sl,x+xx*k,y+yy*k,ww*k,hh*k,s,fill,stroke,size*k,kind,bold)
 def a(xx,yy,xx2,yy2,**kw):return line(sl,x+xx*k,y+yy*k,x+xx2*k,y+yy2*k,**kw)
 shape(sl,x+.12*k,y+.13*k,.36*k,.36*k,MSO_SHAPE.OVAL,NAVY,NAVY);t(.12,.13,.36,.36,'A',15,True,'FFFFFF',PP_ALIGN.CENTER);t(.61,.1,13.5,.43,'AbDev-X overview | Paired sequences → complementary experts → assay predictions',17,True,NAVY)
 t(.23,.65,2.30,.30,'Paired V_{H} / V_{L} input',15,True);b(.23,1.02,2.30,.95,'V_{H}: QVQLVESGG…\nV_{L}: QSVLTQPPS…\n+ HC / LC subtype',fill='FFFFFF',size=13.5)
 # Explicit molecular path: paired sequences -> ABodyBuilder3 -> predicted Fv.
 b(1.08,2.13,1.48,.38,'ABodyBuilder3',fill='E7F1F8',stroke='86ADC7',size=13)
 a(1.82,1.99,1.82,2.11,c='4384A3');a(1.82,2.53,1.82,2.67,c='4384A3')
 for xx,yy,ww,hh,im,lab in [(.28,2.54,.88,.82,395,'IgG: 1IGT'),(1.40,2.69,.86,.75,397,'Predicted Fv')]:
  sl.shapes.add_picture(BytesIO(assets[im]),Inches(x+xx*k),Inches(y+yy*k),width=Inches(ww*k),height=Inches(hh*k));t(xx-.10,3.47,ww+.20,.21,lab,12.5,c=MUTED,align=PP_ALIGN.CENTER)
 t(3.05,.65,2.75,.30,'Five feature blocks',15,True)
 reps=[('ESM-2 650M · 2,560','F8D9DC'),('AbLang2 · 480','E6DDF5'),('Sequence descriptors · 165','DDEECF'),('HC / LC subtype · 5','DCEBF3'),('Structure-informed bio · 57','FCE7AD')]
 for i,(lab,fill) in enumerate(reps):b(3.05,1.06+i*.455,2.75,.37,lab,fill=fill,stroke='B7C8D3',size=14)
 # Fan out sequence/metadata inputs to every applicable nonstructural block.
 a(2.55,1.46,2.74,1.46,arrow=False);a(2.74,1.245,2.74,2.61,arrow=False)
 for i in range(4):a(2.74,1.245+i*.455,3.02,1.245+i*.455)
 # Bio is last, aligned with the predicted Fv for one straight structure-input arrow.
 a(2.28,3.065,3.02,3.065,c='B67B24')
 # A second bus connects all five representations to the expert framework.
 for i in range(5):a(5.82,1.245+i*.455,5.99,1.245+i*.455,arrow=False)
 a(5.99,1.245,5.99,3.065,arrow=False);t(6.24,.65,2.25,.30,'Four experts',15,True)
 for i,lab in enumerate(['E1','A-MGFS','D-Stack','F-MoE']):
  yy=1.08+i*.55;a(5.99,yy+.215,6.21,yy+.215);b(6.24,yy,2.25,.43,lab,BG[i],C[i],14,bold=True);a(8.52,yy+.215,8.83,yy+.215,arrow=False)
 a(8.83,1.295,8.83,2.945,arrow=False);a(8.83,2.12,9.16,2.12)
 b(9.20,1.10,2.45,2.06,'',fill='E8EAF6',stroke='8C98B5')
 t(9.30,1.28,2.25,.30,'Equal-weight fusion',15,True,align=PP_ALIGN.CENTER)
 # Editable stacked fraction: numerator, vinculum, and denominator are native elements.
 t(9.28,1.96,.45,.38,'ŷ_{a} =',14,True,align=PP_ALIGN.CENTER)
 t(9.79,1.81,1.76,.29,'ŷ_{E1} + ŷ_{A} + ŷ_{D} + ŷ_{F}',13.5,True,align=PP_ALIGN.CENTER)
 a(9.78,2.14,11.55,2.14,arrow=False,c=NAVY,lw=1.15)
 t(9.79,2.18,1.76,.35,'4',16,True,align=PP_ALIGN.CENTER)
 t(9.34,2.70,2.18,.29,'w_{a,k} = 0.25',14.5,True,align=PP_ALIGN.CENTER)
 t(12.15,.65,2.80,.30,'10 assays · 6 properties',14.5,True)
 # Property-level output boxes intentionally omit individual assay names.
 categories=[('Hydrophobicity','DDEBF7','5D94BD'),('Self-association','DCF2ED','4E9A87'),('Nonspecific interactions','EBE2F7','9A7CB5'),('Thermal stability','FCEBD7','D39C5A'),('Monomer content','E7EDF2','8198AD'),('Titer / expression','E3F0D7','86AA5D')]
 a(11.70,2.12,11.92,2.12,arrow=False);a(11.92,1.195,11.92,3.245,arrow=False)
 for row,(label,fill,stroke) in enumerate(categories):
  yy=1.03+row*.41
  a(11.92,yy+.165,12.12,yy+.165)
  b(12.15,yy,2.80,.33,label,fill=fill,stroke=stroke,size=14,bold=True)
 t(12.17,3.48,2.73,.20,'5 primary; 5 code-default',12.5,c=MUTED,align=PP_ALIGN.CENTER)
 t(3.05,3.30,8.65,.28,'Bio is structure-informed and includes sequence summaries.',13,c=MUTED)
def note(sl,s):txt(sl,.38,11.64,15.25,.24,s,12.5,c=MUTED)

sl=new('AbDev-X | Overview and expert architectures')
overview(sl,.35,.77,15.3,3.72)
e1(sl,.35,4.68,7.50,3.28);mgfs(sl,8.05,4.68,7.60,3.28)
dstack(sl,.35,8.15,7.50,3.28);fmoe(sl,8.05,8.15,7.60,3.28)
note(sl,'* Shared ridge head: training-fitted standardization; PCA retaining 90% variance if >100 features; RidgeCV. OOF = out-of-fold.')
sl.notes_slide.notes_text_frame.text='Code matched to scripts/aw_moe.py (E1Expert, MGFSExpert, DStackExpert, FMoEExpert) and scripts/final_evaluation.py::pred_awmoe. All diagrams use native editable shapes, connectors, and text. Original molecular illustrations are retained as raster assets. Sequence excerpts are from the existing brazikumab example. The primary model averages four experts. Diagram dimensions refer to the cached operational feature blocks.'
sl.notes_slide.notes_text_frame.text+=' Representation-count clarification: scripts/final_evaluation.py builds five blocks (esm2, ablang2, desc, bio, isotype). The separate historical AbDevFusion in src/abdev/models/fusion.py has six DEFAULT_BASES, but BioFeatures and BioFeatures_lgbm both map to the same bio input in BASE_BLOCK. Six heads are not six distinct feature blocks. Property groups are broad display categories; the ten assay targets remain distinct.'
sl.notes_slide.notes_text_frame.text+=' Feature provenance: the 165 descriptors are computed directly from VH/VL sequences by build_descriptor_matrix. ESM-2 and AbLang2 use frozen sequence embeddings; HC/LC subtype is metadata. The 57-column bio block merges precomputed TAP, MOE_properties, Aggrescan3D, and AntiFold tables. It contains structure-informed quantities and sequence summaries such as CDR Length, so it is not a set of 57 exclusively geometric measurements.'

sl=new('A | Main framework overview','Paired-chain sequence input and five operational representations feed four complementary experts.')
overview(sl,.4,1.35,15.2,3.75)
txt(sl,.65,5.65,14.7,.44,'A shared, regularized prediction head',22,True,NAVY)
for xx,lab,kind in [(1.0,'Training-fitted\nstandardization',MSO_SHAPE.ROUNDED_RECTANGLE),(4.8,'Optional PCA\n90% variance if >100 features',MSO_SHAPE.HEXAGON),(9.0,'RidgeCV\n12 penalties: 10^{-2} to 10^{4}',MSO_SHAPE.ROUNDED_RECTANGLE)]:box(sl,xx,6.45,3.1,1.12,lab,'F3F7FA',LINE,20,kind)
line(sl,4.15,7.01,4.75,7.01);line(sl,7.95,7.01,8.95,7.01)
box(sl,1,8.55,6.5,1.45,'GDPa1 development\n246 antibodies; cluster–isotype CV','EFF4F8',LINE,21)
box(sl,8.1,8.55,6.5,1.45,'GDPa3 external evaluation\n80 antibodies; assay-dependent labels','EFF4F8',LINE,21)
note(sl,'The detailed expert panels follow on slides 3–6. Inputs are paired VH/VL sequences plus HC/LC subtype metadata.')

details=[(e1,'B | E1: select the strongest source','Five source heads compete using inner out-of-fold performance.'),
 (mgfs,'C | A-MGFS: apply an assay-specific feature mask','Assay-related keyword masks select descriptors before a regularized prediction head.'),
 (dstack,'D | D-Stack: combine two prediction streams','Sequence and descriptor/bio streams are combined through an out-of-fold ridge meta-learner.'),
 (fmoe,'E | F-MoE: mix mechanistic group predictions','Eligible groups receive fixed assay-level weights from their development scores.')]
for fn,title,sub in details:
 sl=new(title,sub);fn(sl,.65,2.0,14.7,6.55)
 if fn==e1:foot='The selected source is fixed for each assay. All source scoring and fitting use development data.'
 elif fn==mgfs:foot='* PCA is used only when the masked feature count exceeds 100. Masks with fewer than five features fall back to all descriptor/bio features.'
 elif fn==dstack:foot='The two stream heads are refitted on full GDPa1 after OOF meta-training. The ridge meta-head uses α = 1.'
 else:foot='HIC: hydro + surface | AC-SINS: charge + surface | PR-CHO: charge + CDR + hydro | SMAC: surface + hydro + charge | HAC: plus stability.'
 txt(sl,.85,9.1,14.3,1.13,foot,20,c=MUTED)
 note(sl,'Each expert contributes one prediction to the primary four-expert average. All shapes and text remain editable.')
 sl.notes_slide.notes_text_frame.text='Source implementation: scripts/aw_moe.py. Shared head: StandardScaler, optional PCA(n_components=.9) for >100 features, RidgeCV. For F-MoE, groups with fewer than three matching features are skipped; all-features fallback applies when no group remains. Weights are softmax(max(mean CV Spearman rho,0)/0.1), fixed by assay. Final fusion: equal mean in scripts/final_evaluation.py.'
P.save(OUT)
print(OUT)
print('slides',len(P.slides),'native shapes',sum(len(s.shapes) for s in P.slides))
