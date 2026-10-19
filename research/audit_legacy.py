import pathlib,pickle,json,numpy as np
rows=[]
for p in pathlib.Path('outputs').glob('*/grokking_history.pkl'):
 h=pickle.load(p.open('rb')); row={'run':p.parent.name,'config':h.get('config'),'keys':list(h),'samples':len(h.get('epoch',[]))}
 for k in ['train_acc','val_acc']:
  v=np.asarray(h.get(k,[])); hits=np.flatnonzero(np.convolve((v>=.99).astype(int),np.ones(5,dtype=int),'valid')==5) if len(v)>=5 else []
  row[k]={'last':float(v[-1]) if len(v) else None,'max':float(v.max()) if len(v) else None,'sustained_99_epoch':int(h['epoch'][hits[0]]) if len(hits) else None}
 cfg=h.get('config',{})
 if cfg.get('task','modular')=='modular':
  mod=cfg['modulus']; r=cfg['input_range']; seed=cfg.get('seed',42)
  ids=np.arange(r*r); np.random.default_rng(seed).shuffle(ids); cut=int(len(ids)*cfg['train_fraction'])
  residues=lambda t:set(zip((t//r%mod).tolist(),(t%r%mod).tolist()))
  tr,va=residues(ids[:cut]),residues(ids[cut:])
  row['split_audit']={'identical_token_overlap_within_stage':0,'ordered_residue_pair_overlap':len(tr&va),
   'validation_fraction_residue_pair_seen_in_train':float(np.mean([((i//r%mod,i%r%mod) in tr) for i in ids[cut:]]))}
 row['exponents']=sorted(set(h.get('exponent',[])),key=str); rows.append(row)
pathlib.Path('research/legacy_audit.json').write_text(json.dumps(rows,indent=2))
print(json.dumps([{k:v for k,v in r.items() if k not in ['keys','config']} for r in rows],indent=2))
