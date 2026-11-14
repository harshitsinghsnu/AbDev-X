from pathlib import Path
import re,shutil,json
P=Path(__file__).resolve().parent;OLD=P/'final_bmc_revision';O=P/'abstract_format_revision';O.mkdir(exist_ok=True)
shutil.copytree(OLD/'figures',O/'figures',dirs_exist_ok=True)
text=(OLD/'AbDevX_BMC_updated_main_SI.tex').read_text(encoding='utf8')
abstract=r'''\begin{abstract}
\noindent\textbf{Background:} Early antibody selection requires balancing molecular interaction behavior, stability, and expression. Protein language models and physicochemical descriptors offer complementary information, but combining them does not consistently improve predictions when assay labels are scarce and candidate populations change. We investigated whether assay-aware expert construction could improve external antibody ranking while keeping the final ensemble interpretable.

\noindent\textbf{Results:} AbDev-X combines representation selection, assay-related descriptor masking, prediction stacking, and mechanistic feature-group modeling through a simple equal-weight average. Evaluation using sequence-cluster-aware cross-validation and an independent antibody panel showed stronger aggregate ranking than matched single-representation baselines across the observed positive-signal assays. The clearest improvements concerned hydrophobicity and surface-interaction behavior, supporting candidate prioritization where sequence embeddings alone provide incomplete information. Source-removal analyses revealed assay-dependent contributions from structural bio features and antibody-specific embeddings, explaining why complementary experts can improve prediction. Matched-data comparisons also yielded higher polyreactivity ranking and classification point estimates than the evaluated DELPHI configuration. Primary-model feature decompositions and structure-mapped residue contributions connected predictions to interpretable molecular information while distinguishing model attribution from mechanistic evidence.

\noindent\textbf{Conclusions:} AbDev-X provides an assay-aware framework for combining pretrained sequence representations with explicit molecular descriptors. Its strength lies in selective use of complementary information rather than a universal representation or complex final gate. The results support further evaluation for prioritizing antibodies on hydrophobic and surface-interaction endpoints, with assay-specific validation guiding broader application.
\end{abstract}'''
text=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:abstract,text,count=1,flags=re.S)
text=text.replace(r' source (commit \texttt{231efc9}) with JAX',r' source with JAX')
text=text.replace('This design links the proposed benefit---better use of complementary information in a small-data setting---to both its evidence and its practical limits.','This design tests the proposed benefit of better use of complementary information in a small-data setting, alongside its evidence and practical limits.')
text=text.replace('\u2014',', ').replace('---',', ')
text=text.replace("The designated release location for the AbDev-X source code is the authors' GitHub repository~\\citep{abdevx_repo}.","AbDev-X code and study-specific reproducibility materials are maintained in the authors' GitHub repository and will be made publicly accessible before publication~\\citep{abdevx_repo}.")
text=text.replace('Public access to the AbDev-X repository was not verifiable at the time of this revision; the code release and a release-specific archival identifier remain to be finalized before submission. ','')
text=text.replace(r'AbDev-X: designated source-code repository. \url{https://github.com/harshitsinghsnu/AbDev-X}. Public access not verified on 13 Nov 2026.',r'AbDev-X: source code and reproducibility materials. \url{https://github.com/harshitsinghsnu/AbDev-X}. Public release planned before publication.')
# Add leading zeros only to bare numerical fractions, retaining decimal versions,
# DOI identifiers, asset paths, citation keys, and all numerical values.
text,count=re.subn(r'(?<![A-Za-z0-9])\.(?=\d)','0.',text)
text,math_count=re.subn(r'(\\[A-Za-z]+)\.(?=\d)',lambda m:m.group(1)+'0.',text)
count+=math_count
# Preserve the original table font while allowing the longer leading-zero values.
text=text.replace(r'\footnotesize\setlength{\tabcolsep}{3pt}',r'\footnotesize\setlength{\tabcolsep}{1pt}')
assert '231efc9' not in text
assert '\u2014' not in text and '---' not in text
assert not re.search(r'(?<![A-Za-z0-9])\.\d',text)
visible=abstract.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
assert not re.search(r'\d',visible)
assert not re.search(r'\\texttt\{[^}]*\.(?:py|csv|json|pkl|npz|tex|pdf)',text)
words=len(re.sub(r'\\\w+|[{}]',' ',visible).split());assert words<=350
(O/'AbDevX_BMC_polished_main_SI.tex').write_text(text,encoding='utf8')
(O/'editorial_changes.json').write_text(json.dumps(dict(abstract_words=words,numeric_abstract_results=False,leading_zero_insertions=count,commit_identifiers_removed=True,em_dashes_removed=True,github='https://github.com/harshitsinghsnu/AbDev-X',availability='public release planned before publication; private status not presented as currently public',figure_assets='unchanged'),indent=2))
print('Abstract words:',words,'Leading-zero insertions:',count)
