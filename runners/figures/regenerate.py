"""Generate the numerical result plots without any manuscript dependencies."""
from pathlib import Path
import runpy,sys,os
R=Path(__file__).resolve().parents[2];P=R/'experiments';F=R/'outputs/figures';F.mkdir(parents=True,exist_ok=True);os.chdir(R)
sys.path[:0]=[str(P/'evaluation'),str(P/'plotting'),str(P/'primary'),str(R/'scripts'),str(R/'src')]
import make_revision_figures as m
m.F=F;m.b.F=F
for fn in [m.encoders,m.soft_extensions,m.detailed_shap,m.headline,m.main_context_and_learning]:
 print('Plotting',fn.__name__,flush=True);fn()
m.b.generalization()
for name in ['supplementary_panels.py','baseline_panel.py','bar_panels.py']:
 runpy.run_path(str(Path(__file__).parent/name),run_name='__main__')
try:runpy.run_path(str(Path(__file__).with_name('attribution_panels.py')),run_name='__main__')
except FileNotFoundError as error:print('Cartoon panel requires the ChimeraX render step:',error,flush=True)
print('Numerical plots saved to',F)
