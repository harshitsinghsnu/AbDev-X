from pathlib import Path
base = Path(__file__).with_name('build_architecture_panel.py')
exec(base.read_text(encoding='utf-8').split("text(.5,.22")[0])
prs.slide_width, prs.slide_height = Inches(28), Inches(16.5)
im = Image.new('RGB',(2800,1650),'white'); d=ImageDraw.Draw(im)
BLACK='242424'; GRAY='555555'; LINE='CCCCCC'
text(.5,.65,27,.8,'AbDev-X',37,BLACK,True)
for x,w,s in [(.45,3.95,'(a) Input'),(4.85,5.1,'(b) Representations'),(10.65,13.3,'(c) Four experts + assay-specific fusion'),(24.5,3.1,'(d) Outputs')]:
    text(x,2.05,w,.50,s,21,BLACK)
for x in [4.5,10.3,24.25]:arrow(x,2.05,x,13.0,'D0D0D0',False,.8)

# Editable stylized Fab illustration, following the reference palette.
def rod(x1,y1,x2,y2,c,width=16):arrow(x1,y1,x2,y2,c,False,width)
rod(2.14,5.64,2.14,4.65,'715884',21)
rod(2.14,4.65,1.38,3.65,BLACK,3)
rod(2.14,4.65,2.90,3.65,BLACK,3)
for x1,y1,x2,y2 in [(1.38,3.65,.90,2.99),(1.38,3.65,1.78,3.04),(2.90,3.65,2.50,3.04),(2.90,3.65,3.34,2.99)]:
    rod(x1,y1,x2,y2,'346C92',15)
for x1,y1,x2,y2 in [(.90,2.99,.65,2.64),(1.78,3.04,2.02,2.69),(2.50,3.04,2.26,2.69),(3.34,2.99,3.59,2.64)]:
    rod(x1,y1,x2,y2,'A7CE76',15)
text(.6,5.78,3.2,.35,'Antibody VH + VL',19,BLACK,True)
arrow(2.14,6.26,2.14,6.65,BLACK)
box(.55,6.65,3.4,1.05,'VH sequence\nVL sequence','FFFFFF',BLACK,20,True)
arrow(2.14,7.7,2.14,7.97,BLACK,False)
arrow(2.14,8.36,2.14,8.55,BLACK)
text(.55,8.0,3.4,.3,'ABodyBuilder3',16,BLACK)
asset=OUT/'reference_fv_structure.png'
slide.shapes.add_picture(str(asset),Inches(.6),Inches(8.55),width=Inches(3.25))
pic=Image.open(asset).convert('RGBA'); pic.thumbnail((325,270)); im.paste(pic,(60,855),pic)
text(.45,11.25,3.6,.6,'Predicted Fv structure',18,BLACK)
text(.45,11.88,3.6,.47,'Structure illustration from reference',11,GRAY)

# All six representation blocks retain the reference's saturated color style.
colors=['C54E52','D6C6EB','C5DFE8','59C65B','F4CD45','F5E5A1']
reps=[('ESM-2 650M','Frozen general PLM · pooling · PCA'),('AbLang2','Frozen antibody PLM · pooling · PCA'),('Germline / isotype','Lineage · IgG subclass'),('Physicochemical descriptors','165 sequence-derived features'),('Structural features','SASA · hydrophobic patches · charge'),('Surface biophysics','18 SAP / electrostatic / exposure features')]
for i,(title,sub) in enumerate(reps):
    y=2.95+i*1.5
    box(4.85,y,5.05,1.2,fill=colors[i],stroke=colors[i])
    text(5.02,y+.16,4.71,.38,title,21,BLACK,True)
    text(4.98,y+.67,4.79,.34,sub,14,BLACK)
    arrow(9.9,y+.6,10.10,y+.6,BLACK,False,1)
arrow(10.10,3.55,10.10,11.55,BLACK,False,1)
# Explicit separate sequence and structure routing.
arrow(3.95,7.18,4.22,7.18,BLACK,False,1)
arrow(4.22,3.55,4.22,8.05,BLACK,False,1)
for cy in [3.55,5.05,6.55,8.05]:arrow(4.22,cy,4.85,cy,BLACK,True,1)
arrow(3.85,10.40,4.08,10.40,BLACK,False,1)
arrow(4.08,9.55,4.08,11.05,BLACK,False,1)
for cy in [9.55,11.05]:arrow(4.08,cy,4.85,cy,BLACK,True,1)

# Four expert architectures, each contained within its own corresponding block.
ys=[2.95,5.35,7.75,10.15]
titles=['E1 · Best-block selector','A-MGFS · Assay-masked descriptors','D-Stack · Two-stream ridge stacker','F-MoE · Feature-level mixture']
fills=['FBE9E9','F1EBF8','EAF3F7','FFF5D7']
for y,title,fill in zip(ys,titles,fills):
    box(10.70,y,9.30,2.13,fill=fill,stroke=BLACK)
    text(10.88,y+.10,8.94,.35,title,20,BLACK,True,align='left')
    arrow(10.10,y+1.10,10.70,y+1.10,BLACK,True,1)
    arrow(20,y+1.1,20.38,y+1.1,BLACK,False,1)
arrow(20.38,4.05,20.38,11.25,BLACK,False,1)

def mini(x,y,w,h,label,size=15):return box(x,y,w,h,label,'FFFFFF',BLACK,size)
for y,labels in [(3.62,['Six source\npredictors','OOF scores\nper assay','Best-source\nselection','Refit selected\npredictor']),(6.02,['Descriptor +\nbiophysical inputs','Assay-specific\nfeature mask','Greedy feature\nselection','Assay\nregressor'])]:
    for i,s in enumerate(labels):
        x=10.98+i*2.22; mini(x,y,1.96,.84,s,14)
        if i<3:arrow(x+1.96,y+.42,x+2.22,y+.42,BLACK,True,1)
text(10.99,4.60,8.7,.25,'Select the highest-scoring representation for each assay using GDPa1 OOF predictions.',12,GRAY)
text(10.99,7.00,8.7,.25,'Mechanistic feature masking and greedy selection are performed within training folds.',12,GRAY)

mini(10.98,8.40,2.50,.53,'ESM-2 + AbLang2',14)
mini(10.98,9.11,2.50,.53,'Descriptors + biophysics',13)
for cy in [8.665,9.375]:arrow(13.48,cy,13.78,cy,BLACK,True,1)
mini(13.78,8.40,2.05,.53,'PLM prediction',14)
mini(13.78,9.11,2.05,.53,'Descriptor prediction',13)
for cy in [8.665,9.375]:arrow(15.83,cy,16.18,cy,BLACK,False,1)
arrow(16.18,8.665,16.18,9.375,BLACK,False,1)
arrow(16.18,9.02,16.55,9.02,BLACK,True,1)
mini(16.55,8.62,3.12,.80,'Ridge meta-learner',16)
text(16.55,9.48,3.12,.24,'Nested OOF training',12,GRAY)

mini(10.98,11.03,1.9,.65,'Descriptor\nfeatures',14)
arrow(12.88,11.355,13.17,11.355,BLACK,False,1)
arrow(13.17,10.89,13.17,11.97,BLACK,False,1)
for cy,s in zip([10.89,11.25,11.61,11.97],['Charge','Surface','Hydrophobicity','CDR']):
    arrow(13.17,cy,13.46,cy,BLACK,True,1)
    mini(13.46,cy-.14,2.49,.28,s,11)
    arrow(15.95,cy,16.22,cy,BLACK,False,1)
arrow(16.22,10.89,16.22,11.97,BLACK,False,1)
arrow(16.22,11.355,16.55,11.355,BLACK,True,1)
mini(16.55,10.95,3.12,.80,'Softmax gate\n+ weighted mix',15)
text(16.55,11.87,3.12,.24,'Per-assay gates · τ = 0.1',12,GRAY)

# Learned ensemble weights replace the old single stacking heatmap.
arrow(20.38,7.35,20.82,7.35,BLACK,True,1)
box(20.82,5.65,3.03,3.45,fill='F4F1FA',stroke=BLACK)
text(20.99,5.84,2.69,.80,'Assay-specific\nweighted fusion',19,BLACK,True)
text(20.95,6.90,2.77,.40,'ŷ(a) = Σk w(a,k) ŷk(a)',13,BLACK)
text(20.95,7.50,2.77,.55,'w(a,k) ≥ 0\nΣk w(a,k) = 1',15,BLACK)
text(20.99,8.25,2.69,.58,'GDPa1 OOF weights\nSLSQP optimization',13,BLACK)
arrow(23.85,7.35,24.50,7.35,BLACK,False,1)
arrow(24.50,3.44,24.50,12.08,BLACK,False,1)
assays=['HAC','SMAC','HIC','AC-SINS','PR-CHO','Tm1','Tm2','Tonset','SEC','Titer']
for i,s in enumerate(assays):
    cy=3.44+i*.96
    arrow(24.5,cy,24.92,cy,BLACK,True,1)
    box(24.92,cy-.32,2.48,.64,s,'F1D8A0','F1D8A0',19,True,BLACK,False)
text(24.5,12.57,3.05,.5,'10 continuous\nassay predictions',15,BLACK)

# Low-profile validation strip preserves the reference's uncluttered main flow.
arrow(.55,13.60,27.40,13.60,'D0D0D0',False,.8)
text(.65,13.92,26.7,.44,'GDPa1 (n = 246)  →  5-fold cluster–isotype CV  →  OOF selection & weight fitting  →  Refit four experts  →  GDPa3 held-out (n = 80)',18,BLACK)
text(.65,14.59,26.7,.35,'VH clusters at 70% identity; stratified by IgG subclass. GDPa3 labels excluded from model selection.',15,GRAY)
text(.65,15.15,26.7,.35,'Evaluation: Spearman ρ with 95% BCa bootstrap confidence intervals for held-out metrics.',14,GRAY)
prs.save(OUT/'abdevx_reference_style_updated.pptx')
im.save(OUT/'abdevx_reference_style_updated.png')
print('Created reference-style panel with',len(slide.shapes),'editable objects')
