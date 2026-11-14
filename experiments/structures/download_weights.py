from pathlib import Path
import os,sys,time,json,hashlib,argparse
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
P=Path(__file__).resolve().parent;ap=argparse.ArgumentParser();ap.add_argument('model',choices=['esmfold','af3']);args=ap.parse_args()
sys.stdout.reconfigure(encoding='utf8')
if args.model=='esmfold':
 import requests
 path=P/'weights/ESMFold';path.mkdir(parents=True,exist_ok=True)
 for name in ['config.json','vocab.txt','tokenizer_config.json','special_tokens_map.json','pytorch_model.bin']:
  dest=path/name
  if dest.exists():continue
  response=requests.get('https://huggingface.co/facebook/esmfold_v1/resolve/75a3841ee059df2bf4d56688166c8fb459ddd97a/'+name,stream=True,timeout=(20,90));response.raise_for_status();total=int(response.headers.get('Content-Length',0));got=0;t=time.time();digest=hashlib.sha256()
  with open(path/(name+'.partial'),'wb') as f:
   for chunk in response.iter_content(4*1024*1024):
    f.write(chunk);digest.update(chunk);got+=len(chunk)
    if time.time()-t>20:print(name,'MB',got//1048576,'of',total//1048576,flush=True);t=time.time()
  assert not total or got==total
  if name=='pytorch_model.bin':assert digest.hexdigest()=='2ee07356b125d1e3e57503c204111fd7323347fc4735d41d3caac57c2a78e116'
  (path/(name+'.partial')).replace(dest)
 print('ESMFold weights ready',path,flush=True)
else:
 import requests
 dest=P/'weights/AlphaFold3/af3.bin.zst';dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists() and dest.stat().st_size==1020545840:print('AF3 compressed weights already downloaded');sys.exit()
 response=requests.get('https://storage.googleapis.com/alphafold3/af3.bin.zst',stream=True,timeout=90);response.raise_for_status();total=int(response.headers.get('Content-Length',0));got=0;t=time.time()
 with open(dest.with_suffix('.partial'),'wb') as f:
  for data in response.iter_content(4*1024*1024):
   f.write(data);got+=len(data)
   if time.time()-t>20:print('AF3 MB',got//1048576,'of',total//1048576,flush=True);t=time.time()
 assert not total or got==total
 dest.with_suffix('.partial').replace(dest);print('AF3 weights ready',got,flush=True)
