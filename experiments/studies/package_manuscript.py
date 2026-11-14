"""Prepare a portable, BibTeX-enabled package of the final reviewed manuscript."""
from pathlib import Path
import re,shutil,json,hashlib
P=Path(__file__).resolve().parent;SOURCE=P/'abstract_format_revision';O=P/'AbDevX_BMC_source_package';O.mkdir(exist_ok=True);(O/'figures').mkdir(exist_ok=True)
text=(SOURCE/'AbDevX_BMC_polished_main_SI.tex').read_text(encoding='utf8')
match=re.search(r'\\begin\{thebibliography\}\{99\}.*?\\end\{thebibliography\}',text,re.S);assert match
entries=list(re.finditer(r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})',match.group(),re.S));assert len(entries)==42
fields=[]
for item in entries:
 key,formatted=item.group(1),item.group(2).strip();plain=re.sub(r'\\url\{[^}]*\}\.?','',formatted).strip();urlmatch=re.search(r'\\url\{([^}]+)\}',formatted);url=urlmatch.group(1) if urlmatch else ''
 author,remaining=plain.split('. ',1)
 # The formatted field preserves the current reference style and wording exactly.
 # Standard fields make the database reusable with other bibliography styles.
 article=re.match(r'(.+?)\. ([^.]+(?:\.[^0-9]*)?)\. (\d{4});([^:;.]+):(.+?)\.$',remaining)
 if article:
  title,journal,year,volume,pages=article.groups();kind='article'
 else:
  title=remaining.split('. ',1)[0].rstrip('.');yearmatch=re.search(r'\b(?:19|20)\d{2}\b',remaining);year=yearmatch.group() if yearmatch else '';journal=volume=pages='';kind='misc'
 corp=key in ['moe','delphi','dataset','ginkgo_repo','abdevx_repo']
 if corp:authors='{'+author+'}'
 else:
  names=[]
  for name in author.split(', '):
   if name=='et al':names.append('others');continue
   parts=name.rsplit(' ',1);names.append(parts[0]+', '+parts[1] if len(parts)==2 else name)
  authors=' and '.join(names)
 data={'author':authors,'title':'{'+title+'}'}
 for name,value in [('journal',journal),('year',year),('volume',volume),('pages',pages),('url',url)]:
  if value:data[name]=value
 if url.startswith('https://doi.org/'):data['doi']=url.removeprefix('https://doi.org/')
 data['formatted']=formatted
 fields.append('@'+kind+'{'+key+',\n'+',\n'.join('  '+k+' = {'+v+'}' for k,v in data.items())+'\n}\n')
(O/'references.bib').write_text('% Standard metadata plus exact Vancouver-formatted reference text.\n% The supplied style uses formatted to preserve the approved manuscript.\n\n'+'\n'.join(fields),encoding='utf8')
(O/'bmc_vancouver.bst').write_text(r'''% Preserve the reference wording and numeric citation order of this manuscript.
ENTRY { author title journal year volume pages doi url note formatted } {} {}
FUNCTION {output.entry}
{ newline$ "\bibitem{" write$ cite$ write$ "}" write$ newline$
  formatted empty$ { title write$ } { formatted write$ } if$
  newline$
}
FUNCTION {article} { output.entry }
FUNCTION {misc} { output.entry }
FUNCTION {default.type} { output.entry }
READ
FUNCTION {begin.bib}
{ "\begin{thebibliography}{99}" write$ newline$ }
FUNCTION {end.bib}
{ newline$ "\end{thebibliography}" write$ newline$ }
EXECUTE {begin.bib}
ITERATE {call.type$}
EXECUTE {end.bib}
''',encoding='utf8')
main=text[:match.start()]+r'\bibliographystyle{bmc_vancouver}'+'\n'+r'\bibliography{references}'+text[match.end():]
(O/'main.tex').write_text(main,encoding='utf8')
(O/'main_inline.tex').write_text(text,encoding='utf8')
used=sorted(set(re.findall(r'\\(?:figpanel|widepanel|archpanel)\{([^}]+)\}',text)+re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text)))
used=[name for name in used if not name.startswith('#')]
assert len(used)>15
manifest=[]
for name in used:
 path=SOURCE/'figures'/name;assert path.is_file(),name
 for variant in [path,path.with_suffix('.svg'),path.with_suffix('.png')]:
  if variant.exists():shutil.copy2(variant,O/'figures'/variant.name)
 manifest.append({'figure':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
shutil.copy2(SOURCE/'AbDevX_BMC_polished_main_SI.pdf',O/'AbDevX_BMC_polished_main_SI.pdf')
(O/'package_manifest.json').write_text(json.dumps({'source_version':'AbDevX_BMC_polished_main_SI','figure_count':len(used),'bibliography_entries':len(entries),'main_source':'main.tex','inline_fallback':'main_inline.tex','figures':manifest},indent=2))
(O/'README.txt').write_text('''AbDev-X: complete manuscript and supplementary information source package

Contents
  main.tex                         Main + SI, using references.bib
  references.bib                   All 42 references, with standard metadata
  bmc_vancouver.bst                 Style preserving the manuscript references
  main_inline.tex                  Identical text with an inline bibliography
  AbDevX_BMC_polished_main_SI.pdf    Latest reviewed compiled PDF
  main.pdf                         PDF compiled from the BibTeX-enabled source
  figures/                         Every referenced figure PDF; SVG/PNG variants
                                    are included wherever available
  package_manifest.json            Figure list and SHA-256 checksums

Build from this folder using an installed TeX distribution:
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex

For Overleaf, upload the ZIP and select main.tex as the main document.
No absolute paths or project files outside this folder are required.
The local bibliography style preserves the existing Vancouver reference wording
and citation order using each entry's formatted field. Standard author, title,
year, journal, volume, pages, DOI and URL fields are provided where applicable
for reuse with other styles. If editing reference wording with this style, edit
the formatted field too.

Fallback without BibTeX:
  pdflatex -interaction=nonstopmode -halt-on-error main_inline.tex
  pdflatex -interaction=nonstopmode -halt-on-error main_inline.tex

Both sources contain the complete main paper and supplementary information.
Figure PDFs are the exact assets used in the latest reviewed manuscript.
''',encoding='utf8')
print('Package:',O,'Figures:',len(used),'References:',len(entries))
