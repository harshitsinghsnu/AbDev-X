from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(26), Inches(14.625)
slide = prs.slides.add_slide(prs.slide_layouts[6])
im = Image.new('RGB', (2600,1463), 'white')
d = ImageDraw.Draw(im)
NAVY='19344B'; INK='243746'; GRAY='607581'; LINE='9AAFB9'
TEAL='087F8C'; PURPLE='7758A5'; BLUE='376DB4'; ORANGE='B36B27'
def rgb(h): return RGBColor.from_string(h)
def font(sz,bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/arial'+('bd' if bold else '')+'.ttf',round(sz*100/72))
def text(x,y,w,h,s,size=17,color=INK,bold=False,align='center'):
    sh=slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf=sh.text_frame; tf.clear(); tf.word_wrap=True
    tf.margin_left=tf.margin_right=Inches(.04); tf.margin_top=tf.margin_bottom=0
    tf.vertical_anchor=MSO_ANCHOR.MIDDLE
    lines=s.split('\n')
    for i,line in enumerate(lines):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text=line; p.font.name='Arial'; p.font.size=Pt(size); p.font.bold=bold;p.font.color.rgb=rgb(color)
        p.alignment=PP_ALIGN.CENTER if align=='center' else PP_ALIGN.LEFT
        p.space_before=p.space_after=Pt(0)
    f=font(size,bold); lh=size*100/72*1.16
    yy=y*100+(h*100-len(lines)*lh)/2
    for line in lines:
        tw=d.textlength(line,font=f)
        xx=x*100+(w*100-tw)/2 if align=='center' else x*100+4
        d.text((xx,yy),line,font=f,fill='#'+color); yy+=lh
    return sh
def box(x,y,w,h,label='',fill='FFFFFF',stroke=LINE,size=17,bold=False,color=INK,rounded=True):
    sh=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded: sh.adjustments[0]=.12
    sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill);sh.line.color.rgb=rgb(stroke);sh.line.width=Pt(1.2)
    bounds=(round(x*100),round(y*100),round((x+w)*100),round((y+h)*100))
    d.rounded_rectangle(bounds,radius=12 if rounded else 0,fill='#'+fill,outline='#'+stroke,width=2)
    if label:text(x+.04,y+.03,w-.08,h-.06,label,size,color,bold)
    return sh
def arrow(x1,y1,x2,y2,color=GRAY,head=True,width=1.5):
    sh=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    sh.line.color.rgb=rgb(color);sh.line.width=Pt(width)
    if head:
        end=OxmlElement('a:tailEnd');end.set('type','triangle');sh.line._get_or_add_ln().append(end)
    d.line((x1*100,y1*100,x2*100,y2*100),fill='#'+color,width=max(2,round(width*1.5)))
    if head:
        import math
        a=math.atan2(y2-y1,x2-x1); xx=x2*100; yy=y2*100
        d.polygon([(xx,yy),(xx-10*math.cos(a)+5*math.sin(a),yy-10*math.sin(a)-5*math.cos(a)),(xx-10*math.cos(a)-5*math.sin(a),yy-10*math.sin(a)+5*math.cos(a))],fill='#'+color)
def step(x,y,w,h,s,c=TEAL,fill='FFFFFF',sz=16):return box(x,y,w,h,s,fill,c,sz)

text(.5,.22,25,.6,'AbDev-X | Assay-specific antibody developability prediction',29,NAVY,True)
text(.5,.92,25,.35,'Six complementary representations  →  Four specialist predictors  →  Learned fusion for each assay',17,GRAY)
for x,w,title in [(.45,2.7,'A  INPUT'),(3.6,3.9,'B  REPRESENTATIONS'),(8.1,10.8,'C  SPECIALIST PREDICTORS'),(20,5.5,'D  FUSION & OUTPUT')]:
    text(x,1.55,w,.4,title,20,NAVY,True,align='left')

# Sequence and structure routes are distinct and explicitly wired.
box(.5,3.05,2.5,1.15,'Antibody\nVH + VL sequences','EDF5F8',BLUE,18,True)
arrow(3,3.62,3.3,3.62,BLUE,False)
arrow(3.3,2.7,3.3,6.75,BLUE,False)
for cy in [2.7,4.05,5.4,6.75]:arrow(3.3,cy,3.6,cy,BLUE)
arrow(1.75,4.2,1.75,7.2,BLUE)
text(.65,5.4,1.0,.8,'Structure\nprediction',15,GRAY)
box(.5,7.2,2.5,.8,'ABodyBuilder3','F1F6F8',BLUE,19,True)
arrow(1.75,8,1.75,8.6,BLUE)
box(.5,8.6,2.5,1.1,'Predicted\nFv structure','EDF5F8',BLUE,19,True)
arrow(3,9.15,3.3,9.15,BLUE,False)
arrow(3.3,8.1,3.3,9.45,BLUE,False)
for cy in [8.1,9.45]:arrow(3.3,cy,3.6,cy,BLUE)
reps=[('1  ESM-2 650M','Frozen embeddings · pooling · PCA'),('2  AbLang2','Frozen embeddings · pooling · PCA'),('3  Germline / isotype','One-hot lineage & IgG subclass'),('4  Physicochemical','165 sequence-derived descriptors'),('5  Structural features','SASA · patches · surface charge'),('6  Surface biophysics','18 SAP / electrostatic / exposure')]
for i,(title,sub) in enumerate(reps):
    y=2.2+i*1.35
    box(3.6,y,3.9,1.0,fill='F6F9FB',stroke=BLUE)
    text(3.7,y+.10,3.7,.34,title,19,BLUE,True)
    text(3.68,y+.53,3.74,.30,sub,13,GRAY)
    arrow(7.5,y+.5,7.8,y+.5,BLUE,False)
arrow(7.8,2.7,7.8,10.17,BLUE,False)
text(3.6,10.43,3.9,.65,'Representations are routed\nto the relevant expert inputs.',15,GRAY)

rows=[(2.2,'E1','Best-block selector',TEAL,'F0F8F8'),(4.5,'A-MGFS','Masked greedy selection',PURPLE,'F6F2FA'),(6.8,'D-Stack','Two-stream ridge stacker',BLUE,'F1F5FB'),(9.1,'F-MoE','Feature-level mixture',ORANGE,'FBF5EF')]
for y,title,sub,c,fill in rows:
    box(8.15,y,10.8,2.10,fill=fill,stroke=c)
    text(8.35,y+.17,2.05,.45,title,23,c,True)
    text(8.35,y+.71,2.05,.63,sub.replace(' ','\n',1),15,c)
    arrow(7.8,y+1.05,8.15,y+1.05,c)
    arrow(18.95,y+1.05,19.42,y+1.05,c,False)

# E1: candidate models are scored per assay, then one source is selected.
step(10.7,2.65,1.85,.83,'Six source\npredictors',TEAL)
arrow(12.55,3.065,12.86,3.065,TEAL)
step(12.86,2.65,1.85,.83,'OOF Spearman\nscore per assay',TEAL)
arrow(14.71,3.065,15.02,3.065,TEAL)
step(15.02,2.65,1.75,.83,'Select best\nsource',TEAL)
arrow(16.77,3.065,17.08,3.065,TEAL)
step(17.08,2.65,1.55,.83,'Refit source\npredictor',TEAL)
text(10.7,3.65,7.9,.33,'One selected representation model produces the E1 prediction.',14,GRAY)

# A-MGFS: mechanistic masking followed by feature selection and regression.
step(10.7,4.96,1.85,.83,'Descriptor +\nbiophysical inputs',PURPLE,sz=15)
arrow(12.55,5.375,12.86,5.375,PURPLE)
step(12.86,4.96,1.85,.83,'Assay-specific\nfeature mask',PURPLE)
arrow(14.71,5.375,15.02,5.375,PURPLE)
step(15.02,4.96,1.75,.83,'Greedy feature\nselection',PURPLE)
arrow(16.77,5.375,17.08,5.375,PURPLE)
step(17.08,4.96,1.55,.83,'Assay\nregressor',PURPLE)
text(10.7,5.97,7.9,.33,'Mechanistic subspace selection is performed within training folds.',14,GRAY)

# D-Stack: parallel streams combine through a ridge meta-learner.
step(10.7,7.16,2.65,.58,'ESM-2 + AbLang2',BLUE,sz=16)
step(10.7,8.05,2.65,.58,'Descriptors + biophysics',BLUE,sz=15)
arrow(13.35,7.45,13.66,7.45,BLUE)
arrow(13.35,8.34,13.66,8.34,BLUE)
step(13.66,7.16,1.85,.58,'PLM prediction',BLUE,sz=15)
step(13.66,8.05,1.85,.58,'Descriptor prediction',BLUE,sz=14)
arrow(15.51,7.45,15.86,7.45,BLUE,False)
arrow(15.51,8.34,15.86,8.34,BLUE,False)
arrow(15.86,7.45,15.86,8.34,BLUE,False)
arrow(15.86,7.89,16.2,7.89,BLUE)
step(16.2,7.46,2.43,.86,'Ridge meta-learner',BLUE,sz=16)
text(16.1,8.38,2.6,.27,'Nested OOF training',12,GRAY)

# F-MoE: four feature subgroups have per-assay softmax gates.
step(10.65,9.87,1.47,.70,'Descriptor\nfeatures',ORANGE,sz=15)
arrow(12.12,10.22,12.4,10.22,ORANGE,False)
arrow(12.4,9.58,12.4,10.84,ORANGE,False)
for cy,s in zip([9.58,10.0,10.42,10.84],['Charge','Surface','Hydrophobicity','CDR']):
    arrow(12.4,cy,12.7,cy,ORANGE)
    step(12.7,cy-.17,3.4,.34,s,ORANGE,sz=13)
    arrow(16.1,cy,16.35,cy,ORANGE,False)
arrow(16.35,9.58,16.35,10.84,ORANGE,False)
arrow(16.35,10.22,16.6,10.22,ORANGE)
step(16.6,9.77,2.02,.9,'Softmax gate\n+ weighted mix',ORANGE,sz=15)
text(16.6,10.78,2.02,.25,'Per assay · τ = 0.1',12,GRAY)

# Four scalar expert outputs feed final per-assay nonnegative mixing.
arrow(19.42,3.25,19.42,10.15,NAVY,False)
arrow(19.42,5.98,20,5.98,NAVY)
box(20,4.6,5.45,2.75,fill='EEF6F6',stroke=TEAL)
text(20.18,4.82,5.09,.5,'Assay-specific weighted fusion',22,TEAL,True)
text(20.2,5.45,5.05,.45,'ŷ(a) = Σk w(a,k) ŷk(a)',21,NAVY)
text(20.2,6.02,5.05,.35,'w(a,k) ≥ 0;  Σk w(a,k) = 1',17,NAVY)
text(20.2,6.57,5.05,.52,'Weights fit on GDPa1 OOF predictions\nSLSQP optimization',15,GRAY)
arrow(22.72,7.35,22.72,7.8,TEAL)
box(20,7.8,5.45,3.4,fill='F7FAFB',stroke=TEAL)
text(20.15,7.97,5.15,.4,'10 continuous assay predictions',20,TEAL,True)
for i,s in enumerate(['Aggregation: HAC, SMAC','Hydrophobicity: HIC','Self-association: AC-SINS','Polyreactivity: PR-CHO','Thermostability: Tm1, Tm2, Tonset','Colloidal stability: SEC','Expression: Titer']):
    text(20.30,8.54+i*.335,4.85,.29,s,16,INK,align='left')
text(20.1,2.65,5.2,1.2,'Each expert produces a prediction\nfor each assay. Final weights differ\nby assay and are fixed at inference.',17,GRAY)

# Training strip, separate from inference connections.
box(.5,11.85,24.95,2.25,fill='F5F8FA',stroke=LINE)
text(.75,12.00,3.1,.38,'TRAINING & EVALUATION',17,NAVY,True,align='left')
strip=[(.8,3.0,'GDPa1 · n = 246'),(4.2,4.0,'5-fold cluster–isotype CV'),(8.6,2.7,'OOF predictions'),(11.7,4.7,'Source selection + weight fitting'),(16.8,3.5,'Refit all four experts'),(20.7,4.1,'GDPa3 held-out · n = 80')]
for i,(x,w,s) in enumerate(strip):
    text(x,12.57,w,.43,s,17,NAVY,True)
    if i<len(strip)-1:arrow(x+w+.04,12.79,strip[i+1][0]-.10,12.79,NAVY)
text(.8,13.20,24.0,.34,'VH clusters at 70% identity; stratified by IgG subclass. GDPa3 labels excluded from model selection.',16,GRAY)
text(.8,13.65,24.0,.27,'Evaluation: Spearman ρ with 95% BCa bootstrap confidence intervals for held-out metrics.',14,GRAY)
prs.save(OUT/'abdevx_architecture_expert_details.pptx')
im.save(OUT/'abdevx_architecture_expert_details.png')
print('Created editable panel:',OUT/'abdevx_architecture_expert_details.pptx')
print('Shapes:',len(slide.shapes))
