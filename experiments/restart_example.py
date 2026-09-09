"""A short CPU restart check; this is an optimisation example, not a safety result."""
import copy
from pathlib import Path
import tempfile

import torch
from torch import nn

from alignment_metaplasticity.checkpoint import save_checkpoint, load_checkpoint
from alignment_metaplasticity.repro import set_seed
from alignment_metaplasticity.tasks import make_loader
from alignment_metaplasticity.training import train_epochs


def main():
    set_seed(3)
    initial=nn.Linear(2,2)
    data=(torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]),torch.tensor([1,1,0,0]))
    def setup():
        model=copy.deepcopy(initial)
        return model,torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01),make_loader(data,2,True,17)
    full,fo,fl=setup();part,po,pl=setup()
    scale={n:torch.full_like(p,.5) for n,p in full.named_parameters()}
    kwargs={'device':torch.device('cpu'),'lr':.01,'weight_decay':.01,'update_scale':scale}
    train_epochs(full,fl,epochs=3,optimizer=fo,**kwargs)
    train_epochs(part,pl,epochs=1,optimizer=po,**kwargs)
    provenance={'example':'restart-v1','config':{'lr':.01,'weight_decay':.01},'source':'experiments.restart_example'}
    with tempfile.TemporaryDirectory() as directory:
        path=save_checkpoint(Path(directory)/'epoch.pt',part,po,loaders={'train':pl},
                             protection={'scale':scale},progress={'epoch':1},provenance=provenance)
        restored,ro,rl=setup()
        state=load_checkpoint(path,restored,ro,loaders={'train':rl},expected_provenance=provenance)
        kwargs['update_scale']=state['protection']['scale']
        train_epochs(restored,rl,epochs=3-state['progress']['epoch'],optimizer=ro,**kwargs)
    assert all(torch.equal(p,q) for p,q in zip(full.parameters(),restored.parameters()))
    print('Restored model, AdamW state, protection and shuffled loader reproduce uninterrupted training exactly on CPU.')


if __name__=='__main__':
    main()
