"""Obtain deposited DOI metadata from Crossref; never infer bibliographic fields."""
from pathlib import Path
import json,urllib.request,urllib.parse,concurrent.futures
H=Path(__file__).parent;OUT=H/'references'
DOIS={
'jain2017':'10.1073/pnas.1616408114','raybould2019tap':'10.1073/pnas.1810576116',
'lin2023esm2':'10.1126/science.ade2574','ablang2':'10.1093/bioinformatics/btae618',
'prophet2025':'10.1080/19420862.2025.2593055','ginkgo2026':'10.1080/19420862.2026.2634216',
'jiang2026':'10.1021/acs.jcim.6c02385','anarci':'10.1093/bioinformatics/btv552',
'abb3':'10.1093/bioinformatics/btae576','chennamsetty2009sap':'10.1073/pnas.0904191106',
'freesasa':'10.12688/f1000research.7931.1','prottrans':'10.1109/TPAMI.2021.3095381',
'progen2':'10.1016/j.cels.2023.10.002','igbert':'10.1371/journal.pcbi.1012646',
'antifold':'10.1093/bioadv/vbae202','aggrescan3d':'10.1093/nar/gkv359',
'ridge':'10.1080/00401706.1970.10488634','stacking':'10.1016/S0893-6080(05)80023-1',
'bca':'10.1080/01621459.1987.10478410','xgboost':'10.1145/2939672.2939785',
'tabpfn':'10.1038/s41586-024-08328-6','kyte':'10.1016/0022-2836(82)90515-0',
'eisenberg':'10.1038/299371a0','harris':'10.1021/bi962514+',
}
def fetch(item):
 key,doi=item;dest=OUT/(key+'.json')
 if dest.exists():return key,json.loads(dest.read_text(encoding='utf-8'))
 url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')
 req=urllib.request.Request(url,headers={'User-Agent':'AbDevX manuscript bibliography verification/1.0'})
 try:
  with urllib.request.urlopen(req,timeout=35) as r:record=json.load(r)['message']
  dest.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');return key,record
 except Exception as e:return key,{'error':str(e),'DOI':doi}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:records=dict(pool.map(fetch,DOIS.items()))
 (OUT/'doi_metadata.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 for key,r in records.items():print(key, r.get('title',r.get('error')),r.get('container-title'),r.get('volume'),r.get('page'),r.get('article-number'))
