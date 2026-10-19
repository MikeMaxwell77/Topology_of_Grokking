"""Isolated modular-addition pilot. Training never imports topology packages."""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib
import platform
import json
import random
import numpy as np
import torch
from torch import nn

@dataclass
class Config:
    initialization: str = "pytorch"
    p: int = 113
    width: int = 128
    heads: int = 4
    mlp: int = 512
    seed: int = 0
    fraction: float = .3
    lr: float = .001
    decay: float = 1.

class AdditionTransformer(nn.Module):
    """One causal attention layer, no LayerNorm/dropout/bias, untied output."""
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.p + 1, c.width)
        self.pos = nn.Parameter(torch.randn(3, c.width) / c.width**.5)
        self.qkv = nn.Linear(c.width, 3*c.width, bias=False)
        self.attn_out = nn.Linear(c.width, c.width, bias=False)
        self.mlp_in = nn.Linear(c.width, c.mlp, bias=False)
        self.mlp_out = nn.Linear(c.mlp, c.width, bias=False)
        self.unembed = nn.Linear(c.width, c.p, bias=False)
        self.sites = nn.ModuleDict({k: nn.Identity() for k in
            ['embedding', 'heads', 'attention', 'resid_mid', 'mlp', 'mlp_out', 'resid_post']})

        if c.initialization == 'scaled_normal':
            for param in self.parameters():
                nn.init.normal_(param, std=.8/c.width**.5)

    def forward(self, x):
        c = self.c
        r = self.sites['embedding'](self.embed(x) + self.pos)
        q,k,v = [t.reshape(len(x),3,c.heads,c.width//c.heads).transpose(1,2)
                 for t in self.qkv(r).chunk(3, dim=-1)]
        scores = q @ k.transpose(-1,-2) / (c.width//c.heads)**.5
        scores = scores.masked_fill(torch.ones(3,3,device=x.device,dtype=torch.bool).triu(1), -torch.inf)
        z = self.sites['heads']((scores.softmax(-1) @ v).transpose(1,2))
        a = self.sites['attention'](self.attn_out(z.reshape(len(x),3,c.width)))
        r = self.sites['resid_mid'](r + a)
        h = self.sites['mlp'](self.mlp_in(r).relu())
        out = self.sites['mlp_out'](self.mlp_out(h))
        r = self.sites['resid_post'](r + out)
        return self.unembed(r[:,-1])

    def final_logits(self, x):
        """Equivalent training path: a single layer's output uses only final queries/MLP."""
        c = self.c
        r = self.embed(x) + self.pos
        q,k,v = self.qkv(r).chunk(3,dim=-1)
        q=q[:,-1:].reshape(len(x),1,c.heads,c.width//c.heads).transpose(1,2)
        k=k.reshape(len(x),3,c.heads,c.width//c.heads).transpose(1,2)
        v=v.reshape(len(x),3,c.heads,c.width//c.heads).transpose(1,2)
        pattern=(q@k.transpose(-1,-2)/(c.width//c.heads)**.5).softmax(-1)
        z=(pattern@v).transpose(1,2).reshape(len(x),c.width)
        mid=r[:,-1]+self.attn_out(z)
        return self.unembed(mid+self.mlp_out(self.mlp_in(mid).relu()))

def grid(c):
    a,b = torch.meshgrid(torch.arange(c.p),torch.arange(c.p),indexing='ij')
    return torch.stack([a.flatten(),b.flatten(),torch.full((c.p*c.p,),c.p)],1), ((a+b)%c.p).flatten()

def split(c):
    ids = torch.randperm(c.p*c.p, generator=torch.Generator().manual_seed(c.seed))
    n = int(c.fraction*len(ids))
    if not 0 < n < len(ids):
        raise ValueError('empty split')
    return ids[:n], ids[n:]

def metrics(logits,y):
    return {'loss': nn.functional.cross_entropy(logits.double(),y).item(),
            'accuracy': (logits.argmax(-1)==y).double().mean().item()}

def sustained(rows,key, threshold=.99, count=5):
    for i in range(len(rows)-count+1):
        if all(r[key]['accuracy'] >= threshold for r in rows[i:i+count]):
            return rows[i]['step']
    return None

def behavior_timing(rows):
    memorization = sustained(rows,'train')
    generalization = sustained(rows,'heldout')
    plateau = [r for r in rows if memorization is not None and r['step'] >= memorization
               and (generalization is None or r['step'] < generalization)]
    low = sustained([dict(r,low={'accuracy':float(r['heldout']['accuracy']<.5)}) for r in plateau], 'low')
    delayed = (memorization is not None and generalization is not None and
               generalization-memorization >= 500 and low is not None)
    return {'memorization':memorization,'generalization':generalization,
            'delayed_generalization':delayed,
            'rule':'accuracy >= .99 for five logged evaluations; delayed requires >=500 steps lag and five heldout evaluations below .5 after memorization'}

def train(args):
    c = Config(p=args.p,seed=args.seed,initialization=args.initialization)
    random.seed(c.seed); np.random.seed(c.seed); torch.manual_seed(c.seed)
    torch.set_num_threads(args.threads)
    resume = torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    if resume:
        c = Config(**resume['config'])
    device = torch.device(args.device)
    m = AdditionTransformer(c).to(device)
    opt = torch.optim.AdamW(m.parameters(),lr=c.lr,weight_decay=c.decay,betas=(.9,.98))
    x,y = [t.to(device) for t in grid(c)]
    tr,va = [t.to(device) for t in split(c)]
    probe = torch.randperm(len(x),generator=torch.Generator().manual_seed(c.seed+100))[:args.probes]
    start = 0
    rows = []
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if resume:
        start = resume['step']
        if args.steps <= start:
            raise ValueError('steps must exceed resume checkpoint step')
        m.load_state_dict(resume['model']); opt.load_state_dict(resume['optimizer'])
        tr,va = resume['train'].to(device),resume['heldout'].to(device)
        probe = resume['probe']
        torch.set_rng_state(resume['torch_rng']); np.random.set_state(resume['numpy_rng'])
        random.setstate(resume['python_rng'])
        if resume['cuda_rng'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        old = Path(args.resume).parent/'behavior.json'
        if old.exists():
            rows = [r for r in json.loads(old.read_text()) if r['step'] < start]
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    (root/'config.json').write_text(json.dumps(asdict(c),indent=2))
    for step in range(start,args.steps+1):
        if step % args.log_every == 0 or step == args.steps:
            m.eval()
            with torch.no_grad():
                logits=torch.cat([m.final_logits(b) for b in x.split(1024)])
                rows.append({'step':step,'train':metrics(logits[tr],y[tr]),'heldout':metrics(logits[va],y[va])})
            (root/'behavior.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(rows[-1]),flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'config':asdict(c),
                'schema_version':1,'source_sha256':source_hash,'python_version':platform.python_version(),
                'grid_definition':'lexicographic a,b in range(p); inputs [a,b,p]; target (a+b)%p',
                'step':step,'train':tr.cpu(),'heldout':va.cpu(),'probe':probe,
                'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
                'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                'torch_version':torch.__version__},root/f'step_{step:06d}.pt')
        if step == args.steps: break
        m.train(); opt.zero_grad()
        loss=nn.functional.cross_entropy(m.final_logits(x[tr]).double(),y[tr]); loss.backward(); opt.step()
    (root/'timing.json').write_text(json.dumps(behavior_timing(rows),indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--resume',help='trusted local checkpoint; continue into a new output directory')
    p.add_argument('--output',required=True); p.add_argument('--steps',type=int,default=40000)
    p.add_argument('--initialization',choices=['pytorch','scaled_normal'],default='scaled_normal')
    p.add_argument('--p',type=int,default=113); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--log-every',type=int,default=50); p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--probes',type=int,default=128); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if min(args.steps,args.log_every,args.save_every,args.probes,args.threads) <= 0 or args.p<3 or args.p%2==0:
        p.error('positive intervals/counts and odd p >=3 required')
    train(args)
