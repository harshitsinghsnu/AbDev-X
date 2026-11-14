from pathlib import Path
from pptx import Presentation
from pptx.util import Pt, Inches
from pptx.enum.text import MSO_ANCHOR
H=Path(__file__).parent
p=Presentation(H/'architecture.pptx')
replacements={151:'5 source\nheads',157:'CV scores',163:'Select best\nsource',169:'Refit ridge',174:'Desc + bio',180:'Assay mask',186:'Scale +\nPCA',192:'Ridge head',195:'Five operational blocks; select by inner OOF score.',198:'Keyword masks; preprocessing fitted on training data.',203:'ESM + AbL',208:'Desc + bio',215:'PLM head',220:'Bio head',229:'OOF ridge stack',232:'Two prediction streams',237:'Desc + bio',259:'Hydrophobic',274:'CV softmax\nweighted sum',277:'Fixed by assay; tau = 0.1',369:'Four-expert\naverage',372:'Each weight\n= 0.25',392:'Equal-weight average',393:'Fixed at inference',396:'Illustrative IgG: PDB 1IGT',402:'Heavy-chain VH',403:'QVQLVESGGGVVQ...',404:'Light-chain VL',405:'QSVLTQPPSVSGA...',60:'Frozen PLM; pooled VH/VL',69:'Frozen antibody PLM',78:'IgG subclass; kappa/lambda',87:'165 sequence features',96:'57 merged bio features',105:'Within the same bio block',52:'Brazikumab; predicted Fv',357:'10 assay\npredictions',84:'Sequence descriptors',13:'(c) Four experts + equal-weight fusion'}
replacements.update({10:'(b) Features',13:'(c) Four experts and fusion',232:'',277:'',398:'',399:'',408:'',401:'Sequence input',402:'VH sequence',403:'QVQLVESGG...',404:'VL sequence',405:'QSVLTQPPS...',369:'4-expert mean',372:'w = 0.25',392:'Equal weights',393:'Fixed inference',52:'Brazikumab Fv',259:'Hydrophobic'})
def walk(shapes):
 for s in shapes:
  if s.shape_type==6:walk(s.shapes)
  elif s.has_text_frame and s.text.strip():
   if s.shape_id in replacements:s.text=replacements[s.shape_id]
   if not s.text.strip():continue
   tf=s.text_frame;tf.word_wrap=True;tf.vertical_anchor=MSO_ANCHOR.MIDDLE
   tf.margin_top=tf.margin_bottom=0;tf.margin_left=tf.margin_right=Inches(.015)
   # Native slide is very wide: 22 pt becomes approximately 5.6 pt at 183 mm.
   for para in tf.paragraphs:
    para.space_before=para.space_after=Pt(0)
    for r in para.runs:
     r.font.name='Arial';r.font.size=Pt(23 if s.height<Inches(.5) else 25)
     if s.shape_id in [7,10,13,16,122,129,136,143]:r.font.size=Pt(26);r.font.bold=True
   # Give short annotations room for a legible line without moving connectors.
   if s.height<Inches(.31):s.height=Inches(.34)
   if s.shape_id in [401,402,403,404,405]:
    s.height=Inches(.42)
    s.top=Inches({401:6.02,402:6.65,403:7.03,404:7.60,405:8.00}[s.shape_id])
    for para in tf.paragraphs:
     for r in para.runs:r.font.size=Pt(22)
for slide in p.slides:walk(slide.shapes)
for sid in list(p.slides._sldIdLst)[1:]:
 p.part.drop_rel(sid.rId);p.slides._sldIdLst.remove(sid)
p.save(H/'AbDevX_architecture_code_matched.pptx')
print('Saved code-matched native architecture; original shapes and molecular images retained.')
