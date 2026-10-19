import unittest
import numpy as np
import torch
from grokking.circuits import Config, AdditionTransformer, grid, split, sustained
from grokking.circuit_analysis import capture, spectrum, normalized, topology, compare

class CircuitTests(unittest.TestCase):
    def test_unique_split(self):
        c=Config(p=7); x,y=grid(c); a,b=split(c)
        self.assertEqual(len(torch.unique(x,dim=0)),49)
        self.assertFalse(set(a.tolist())&set(b.tolist()))
        self.assertTrue(torch.equal(y,(x[:,0]+x[:,1])%7))
    def test_sum_and_difference_spectrum(self):
        c=Config(p=7); x,y=grid(c)
        for values,key in [(y,'sum_fraction'),(x[:,0]-x[:,1],'difference_fraction'),(x[:,0],'a_fraction')]:
            z=torch.cos(2*torch.pi*values.double()/7)[:,None]
            self.assertAlmostEqual(spectrum(z,7)[key],1.)
    def test_training_path_matches_full_forward_and_gradients(self):
        m=AdditionTransformer(Config(p=7,width=16,heads=4,mlp=32))
        x,y=grid(m.c)
        full=m(x); loss=torch.nn.functional.cross_entropy(full,y); loss.backward()
        grads=[p.grad.clone() for p in m.parameters()]
        m.zero_grad(); fast=m.final_logits(x)
        torch.nn.functional.cross_entropy(fast,y).backward()
        self.assertTrue(torch.allclose(full,fast,atol=1e-6))
        for expected,p in zip(grads,m.parameters()):
            self.assertTrue(torch.allclose(expected,p.grad,atol=1e-6))

    def test_hooks_and_causality(self):
        m=AdditionTransformer(Config(p=7,width=16,heads=4,mlp=32)).eval()
        x,_=grid(m.c); out,c=capture(m,x)
        self.assertEqual(c['mlp'].shape,(49,3,32))
        changed=x.clone(); changed[:,1]=(changed[:,1]+1)%7
        _,d=capture(m,changed)
        self.assertTrue(torch.equal(c['attention'][:,0],d['attention'][:,0]))
        handle=m.sites['mlp'].register_forward_hook(lambda mod,inp,out: torch.zeros_like(out))
        self.assertFalse(torch.equal(out,m(x))); handle.remove()
    def test_real_topology_and_identity_intervention(self):
        t=np.linspace(0,2*np.pi,24,endpoint=False)
        points=np.stack([np.cos(t),np.sin(t)],1)
        d=topology(points,[0.,.5,2.])
        self.assertEqual(d[0]['infinite_count'],1)
        self.assertEqual(d[1]['betti_curve'],[0,1,0])
        self.assertGreater(d[1]['max_persistence'],1.)
        self.assertTrue(np.allclose(compare(d,d),[0,0]))
        m=AdditionTransformer(Config(p=7,width=16,heads=4,mlp=32)).eval()
        x,_=grid(m.c); logits,cache=capture(m,x)
        handle=m.sites['mlp'].register_forward_hook(lambda mod,inp,out: cache['mlp'])
        self.assertTrue(torch.equal(logits,m(x))); handle.remove()

    def test_scale_and_sustained_rule(self):
        z=np.random.default_rng(1).normal(size=(20,5))
        self.assertTrue(np.allclose(normalized(z)[0],normalized(5*z+2)[0]))
        rows=[{'step':i,'train':{'accuracy':a}} for i,a in enumerate([1,1,0,1,1,1,1,1])]
        self.assertEqual(sustained(rows,'train'),3)

if __name__=='__main__': unittest.main()
