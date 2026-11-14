"""Apply the supplied BMC research-article section order; preserve result assets."""
from pathlib import Path
import re,shutil,json,hashlib
P=Path(__file__).resolve().parent;OUT=P.parent/'bmc_formatted_revision';OUT.mkdir(exist_ok=True)
shutil.copytree(P/'figures',OUT/'figures',dirs_exist_ok=True)
text=(P/'AbDevX_structure_comparison_main_SI.tex').read_text(encoding='utf8')
remove='The paired difference intervals included zero. Across all ten endpoints, mean correlation was $.249$; Tm2 and SEC remained weaknesses, including against published reference points.'
assert text.count(remove)==1;text=text.replace(remove,'')
for heading in ['Background','Results','Conclusions']:text=text.replace(r'\noindent\textbf{'+heading+'.}',r'\noindent\textbf{'+heading+':}')
text=text.replace(r'\maketitle',r'\maketitle'+'\n'+r'\thispagestyle{empty}\clearpage',1)
starts={k:text.index(r'\section{'+k+'}') for k in ['Background','Results','Discussion','Conclusions','Methods']}
declarations_start=text.index(r'\section*{Abbreviations}')
prefix=text[:starts['Background']]
background=text[starts['Background']:starts['Results']]
results=text[starts['Results']:starts['Discussion']]
discussion=text[starts['Discussion']:starts['Conclusions']]
conclusions=text[starts['Conclusions']:starts['Methods']]
methods=text[starts['Methods']:declarations_start]
tail=text[declarations_start:]
# The previous draft included an empty subsection immediately before another heading.
methods=methods.replace(r'\subsection{Interpretation and comparison analyses}'+'\n','')
methods+=r'''
\subsection{Use of generative artificial intelligence}
OpenAI ChatGPT/Codex assisted with manuscript editing and organization, code development, figure preparation, and execution of computational workflows. Numerical results were derived from the recorded model predictions and analysis outputs rather than generated as narrative estimates. Generated text and code were checked against the available source files and execution records. These tools are not listed as authors; responsibility for the scientific content and final manuscript rests with the authors.

'''
text=prefix+background+methods+results+discussion+conclusions+tail
text=text.replace(r'\section*{Abbreviations}',r'\section*{List of abbreviations}')
names=['Ethics approval and consent to participate','Consent for publication','Availability of data and materials','Competing interests','Funding',"Authors' contributions",'Acknowledgements']
for name in names:
 old=r'\paragraph{'+name+'.} '
 assert text.count(old)==1,name
 text=text.replace(old,r'\subsection*{'+name+'}\n'.replace('\\n','\n'))
text=text.replace('The authors declare no competing interests.','The authors declare that they have no competing interests.')
# Vancouver numbering follows first citation after the section-order change.
begin=text.index(r'\begin{thebibliography}{99}');end=text.index(r'\end{thebibliography}',begin)+len(r'\end{thebibliography}')
bib=text[begin:end];entries={m.group(1):m.group(0).strip() for m in re.finditer(r'\\bibitem\{([^}]+)\}.*?(?=\\bibitem|\\end\{thebibliography\})',bib,re.S)}
body=text[:begin]+text[end:];order=[]
for m in re.finditer(r'\\cite(?:p|t)?(?:\[[^\]]*\])*\{([^}]+)\}',body):
 for key in m.group(1).split(','):
  key=key.strip()
  if key not in entries:raise ValueError('Missing bibliography entry: '+key)
  if key not in order:order.append(key)
order += [k for k in entries if k not in order]
newbib=r'\begin{thebibliography}{99}'+'\n'+'\n'.join(entries[k] for k in order)+'\n'+r'\end{thebibliography}'
text=text[:begin]+newbib+text[end:]
abstract=text.split(r'\begin{abstract}',1)[1].split(r'\end{abstract}',1)[0]
plain=re.sub(r'\\[a-zA-Z]+(?:\[[^\]]*\])?',' ',abstract);plain=re.sub(r'[{}$]','',plain)
wordcount=len(plain.split());assert wordcount<=350,wordcount
sections=re.findall(r'\\section\*?\{([^}]+)\}',text)
assert sections[:8]==['Background','Methods','Results','Discussion','Conclusions','List of abbreviations','Declarations','Supplementary Information'],sections
(OUT/'AbDevX_BMC_Bioinformatics_main_SI.tex').write_text(text,encoding='utf8')
manifest={'format_source':'User-supplied BMC Bioinformatics Research article requirements','section_order':sections,'abstract_words_approx':wordcount,'declarations_subheadings':names,'reference_count':len(entries),'removed':'Two specified sentences from the abstract; detailed results and uncertainty retained in Results, Discussion and SI','existing_figure_assets_changed':False,'new_disclosure':'Methods: Use of generative artificial intelligence','previous_source':str(P/'AbDevX_structure_comparison_main_SI.tex')}
(OUT/'format_check.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
