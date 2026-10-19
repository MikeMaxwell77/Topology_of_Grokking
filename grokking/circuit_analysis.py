"""Offline activation, Fourier, topology and controlled-intervention analysis."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from .circuits import AdditionTransformer, Config, grid, metrics

def capture(model,x):
    cache={}; handles=[]
    for name,site in model.sites.items():
        handles.append(site.register_forward_hook(lambda mod,inp,out,k=name: cache.__setitem__(k,out.detach().clone())))
    try:
        logits=model(x)
    finally:
        for h in handles: h.remove()
    return logits,cache

def spectrum(z,p):
    z=z.reshape(p,p,-1).double().cpu().numpy()
    z=z-z.mean((0,1),keepdims=True)
    energy=np.abs(np.fft.fft2(z,axes=(0,1),norm='ortho'))**2
    power=energy.sum(-1); total=power.sum()
    den=total if total else 1.
    return {'a_fraction':float(power[1:,0].sum()/den),'b_fraction':float(power[0,1:].sum()/den),
            'sum_fraction':float(np.trace(power)[()]/den),
            'difference_fraction':float(sum(power[k,(-k)%p] for k in range(1,p))/den),
            'power':power.tolist(),'variance':float(total/(p*p))}

def topology(z, scales):
    from ripser import ripser
    z=np.asarray(z,dtype=np.float64)
    from scipy.spatial.distance import pdist, squareform
    diagrams=ripser(squareform(pdist(z)),maxdim=1,distance_matrix=True)['dgms']
    result=[]
    for d in diagrams:
        finite=d[np.isfinite(d).all(1)]
        life=finite[:,1]-finite[:,0]
        result.append({'diagram':d.tolist(),'finite_count':len(finite),'infinite_count':int((~np.isfinite(d).all(1)).sum()),
            'total_persistence':float(life.sum()),'max_persistence':float(life.max()) if len(life) else 0.,
            'betti_curve':[int(((d[:,0]<=s)&(d[:,1]>s)).sum()) for s in scales]})
    return result

def normalized(z):
    z=z-z.mean(0,keepdims=True)
    scale=np.sqrt(np.square(z).sum(1).mean())
    return z/(scale if scale else 1.),float(scale)

def compare(first,second):
    from persim import wasserstein
    values=[]
    for a,b in zip(first,second):
        a=np.array(a['diagram']).reshape(-1,2); b=np.array(b['diagram']).reshape(-1,2)
        values.append(float(wasserstein(a[np.isfinite(a).all(1)],b[np.isfinite(b).all(1)])))
    return values

def run(args):
    torch.set_num_threads(args.threads)
    if 'mlp.equals' not in args.topology_sites.split(','):
        raise ValueError('primary mlp.equals topology site is required')
    root=Path(args.run); dest=Path(args.output) if args.output else root/'analysis'; dest.mkdir(parents=True,exist_ok=False)
    rows=[]; previous={}; scales=np.linspace(0,3,61)
    paths=sorted(root.glob('step_*.pt'))[::args.stride]
    if args.steps:
        selected=[int(step) for step in args.steps.split(',')]
        if selected != sorted(set(selected)):
            raise ValueError('checkpoint steps must be strictly increasing')
        paths=[root/f'step_{step:06d}.pt' for step in selected]
    if not paths:
        raise ValueError('no checkpoints found')
    if args.candidate_frequencies:
        fixed_keys=[int(k) for k in args.candidate_frequencies.split(',')]
    else:
        selection=torch.load(paths[-1],map_location='cpu',weights_only=False)
        selection_model=AdditionTransformer(Config(**selection['config']))
        selection_model.load_state_dict(selection['model']); selection_model.eval()
        with torch.no_grad():
            _,selection_cache=capture(selection_model,grid(selection_model.c)[0])
        selection_power=np.asarray(spectrum(selection_cache['mlp'][:,-1],selection_model.c.p)['power'])
        fixed_keys=(np.argsort(np.diag(selection_power)[1:selection_model.c.p//2+1])[-args.frequencies:]+1).tolist()
        del selection_cache,selection_model
    for path in paths:
        # Only load checkpoints produced by this trusted local pipeline.
        saved=torch.load(path,map_location='cpu',weights_only=False)
        c=Config(**saved['config']); p=c.p
        m=AdditionTransformer(c); m.load_state_dict(saved['model']); m.eval()
        x,y=grid(c); probe=saved['probe'].numpy(); held=saved['heldout']
        with torch.no_grad(): logits,cache=capture(m,x)
        row={'step':saved['step'],'train':metrics(logits[saved['train']],y[saved['train']]),
            'heldout':metrics(logits[held],y[held]),'features':{},'topology':{},'interventions':[]}
        for name,z in cache.items():
            if name=='heads': continue
            for pos,label in enumerate(['a','b','equals']):
                key=f'{name}.{label}'; data=z[:,pos].numpy()
                row['features'][key]=spectrum(z[:,pos],p)
                if key not in args.topology_sites.split(','): continue
                sample=data[probe]; unit,scale=normalized(sample)
                for mode,points in [('raw',sample),('rms',unit)]:
                    keymode=f'{key}.{mode}'; topo=topology(points,scales)
                    row['topology'][keymode]={'dimensions':topo,'rms_scale':scale,'n':len(points),'nested_half':topology(points[:len(points)//2],scales),
                        'wasserstein_previous':compare(previous[keymode],topo) if keymode in previous else None}
                    previous[keymode]=topo
        # Weight periodicity across input residues and output classes.
        for name,w in [('embedding',m.embed.weight[:p]),('unembedding',m.unembed.weight),('neuron_logit',m.unembed.weight @ m.mlp_out.weight)]:
            w=w.detach().numpy(); power=(np.abs(np.fft.fft(w-w.mean(0),axis=0))**2).sum(1)
            row['features'][name]={'frequency_power':power.tolist()}
        h=cache['mlp'][:,-1].double()
        # Explore strongest sum-related sine/cosine pairs, selected without held-out labels.
        fullpower=np.asarray(row['features']['mlp.equals']['power'])
        keys=fixed_keys
        if len(keys)!=len(set(keys)) or any(k<1 or k>p//2 for k in keys):
            raise ValueError('unique candidate frequencies in 1..p//2 required')
        row['candidate_frequencies']=keys
        groups=[[k] for k in keys]+([keys] if len(keys)>1 else [])
        basepred=logits.argmax(-1)
        predictions={'clean':basepred.numpy(),'targets':y.numpy(),'heldout_indices':held.numpy()}
        for group in groups:
            t=2*torch.pi*y.double()/p
            basis=torch.stack([f(k*t) for k in group for f in [torch.cos,torch.sin]],1)
            q=torch.linalg.qr(basis).Q; removed=q@(q.T@h)
            for control in range(-1,args.controls):
                if control<0:
                    delta=removed; name='sum_'+','.join(map(str,group))
                else:
                    gen=torch.Generator().manual_seed(c.seed+1000+control+sum(group))
                    qr=torch.linalg.qr(torch.randn(len(x),basis.shape[1],generator=gen,dtype=torch.float64)).Q
                    delta=qr@(qr.T@h)
                    delta=delta*(removed.norm()/delta.norm().clamp_min(1e-30))
                    name='random_'+','.join(map(str,group))+f'_{control}'
                replacement=(h-delta).float()
                def hook(mod,inp,out):
                    out=out.clone(); out[:,-1]=replacement; return out
                handle=m.sites['mlp'].register_forward_hook(hook)
                try:
                    with torch.no_grad(): changed=m(x)
                finally: handle.remove()
                points,scale=normalized(replacement.numpy()[probe]); topo=topology(points,scales)
                clean=row['topology']['mlp.equals.rms']['dimensions']
                row['interventions'].append({'name':name,'rank':basis.shape[1],
                    'removed_norm':float(delta.norm()),'heldout':metrics(changed[held],y[held]),
                    'prediction_change_rate':float((changed[held].argmax(-1)!=basepred[held]).float().mean()),
                    'topology':topo,'topology_distance_from_clean':compare(clean,topo),
                    'raw_topology':topology(replacement.numpy()[probe],scales)})
                predictions[name]=changed.argmax(-1).numpy()
        # Head interventions at the same checkpoint; all single heads and pairs.
        for size in [1,2]:
            for group in itertools.combinations(range(c.heads),size):
                def hook(mod,inp,out):
                    out=out.clone(); out[:,:,list(group)]=0; return out
                handle=m.sites['heads'].register_forward_hook(hook)
                try:
                    with torch.no_grad(): changed=m(x)
                finally: handle.remove()
                row['interventions'].append({'name':f'heads_{group}','heldout':metrics(changed[held],y[held])})
                predictions[f'heads_{group}']=changed.argmax(-1).numpy()
        np.savez_compressed(dest/f'predictions_{saved["step"]:06d}.npz',**predictions)
        (dest/f'measurements_{saved["step"]:06d}.json').write_text(json.dumps(row))
        rows.append(row); print('analyzed',saved['step'],flush=True)
    (dest/'measurements.json').write_text(json.dumps({'scales':scales.tolist(),'rows':rows},indent=2))
    plot(rows,dest)

def plot(rows,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,sharex=True,figsize=(10,12)); steps=[r['step'] for r in rows]
    for split in ['train','heldout']:
        axes[0].plot(steps,[r[split]['accuracy'] for r in rows],label=split)
        axes[1].plot(steps,[r[split]['loss'] for r in rows],label=split)
    for mode in ['raw','rms']:
        axes[2].plot(steps,[r['topology'][f'mlp.equals.{mode}']['dimensions'][1]['total_persistence'] for r in rows],label=mode)
    for feature in ['a_fraction','b_fraction','sum_fraction','difference_fraction']:
        axes[3].plot(steps,[r['features']['mlp.equals'][feature] for r in rows],label=feature)
    for ax,label in zip(axes,['accuracy','cross entropy','H1 total persistence','Fourier energy fraction']):
        ax.set_ylabel(label); ax.legend()
    axes[-1].set_xlabel('optimizer step'); fig.tight_layout(); fig.savefig(dest/'aligned.png'); fig.savefig(dest/'aligned.svg'); plt.close(fig)
    fig,axes=plt.subplots(2,1,sharex=True,figsize=(10,8))
    names=[i['name'] for i in rows[-1]['interventions'] if not i['name'].startswith('heads')]
    for name in names:
        found=[next(i for i in r['interventions'] if i['name']==name) if any(i['name']==name for i in r['interventions']) else None for r in rows]
        axes[0].plot(steps,[i['heldout']['loss']-r['heldout']['loss'] if i else np.nan for i,r in zip(found,rows)],label=name)
        axes[1].plot(steps,[i['topology_distance_from_clean'][1] if i else np.nan for i in found],label=name)
    axes[0].set_ylabel('heldout loss increase'); axes[1].set_ylabel('H1 Wasserstein from clean')
    axes[1].set_xlabel('optimizer step'); axes[0].legend(fontsize=7,ncol=3)
    fig.tight_layout(); fig.savefig(dest/'interventions.svg'); plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__); p.add_argument('run'); p.add_argument('--output'); p.add_argument('--steps',help='comma-separated checkpoint steps in increasing order'); p.add_argument('--stride',type=int,default=1)
    p.add_argument('--candidate-frequencies',help='frozen frequencies; otherwise select from last analyzed checkpoint')
    p.add_argument('--frequencies',type=int,default=2); p.add_argument('--controls',type=int,default=3)
    p.add_argument('--topology-sites',default='mlp.equals,resid_post.equals,attention.equals,embedding.a,embedding.b,mlp_out.equals')
    p.add_argument('--threads',type=int,default=4); args=p.parse_args()
    if min(args.stride,args.frequencies,args.controls,args.threads)<=0: p.error('positive settings required')
    run(args)
