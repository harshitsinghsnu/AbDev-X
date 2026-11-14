from pathlib import Path
import requests,json,concurrent.futures
H=Path(__file__).parent/'references'
URLS={
 'amini.xml':'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13007433/fullTextXML',
 'jiang_si.pdf':'https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.6c02385/suppl_file/ci6c02385_si_001.pdf',
 'jiang_article.html':'https://pubs.acs.org/doi/full/10.1021/acs.jcim.6c02385',
 'amini_crossref.json':'https://api.crossref.org/works/10.1080/19420862.2026.2647489',
}
def get(kv):
 n,u=kv
 try:
  r=requests.get(u,timeout=35,headers={'User-Agent':'Mozilla/5.0'});(H/n).write_bytes(r.content);return n,r.status_code,len(r.content),r.headers.get('Content-Type')
 except Exception as e:return n,str(e)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for out in pool.map(get,URLS.items()):print(out,flush=True)
try:
 r=requests.post('https://api.figshare.com/v2/articles/search',json={'resource_doi':'10.1021/acs.jcim.6c02385'},timeout=35);(H/'figshare_search.json').write_text(r.text);print('figshare',r.status_code,[(x.get('id'),x.get('title')) for x in r.json()][:3])
except Exception as e:print(e)
