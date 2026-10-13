"""Held-out conditional-vector-field MSE (diagnostic, not a visual quality benchmark)."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from iconflow.train import VelocityUNet,make_icon,flow_interpolant
import numpy as np

def evaluate(checkpoint,n=80,seed=454,size=None,batch=16):
    torch.manual_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved=torch.load(checkpoint,map_location=device,weights_only=True)
    size=int(size or saved['size'])
    model=VelocityUNet(base=int(saved['width'])).to(device)
    model.load_state_dict(saved['model']);model.eval()
    tot,baseline=0.,0.
    with torch.no_grad():
        for first in range(0,n,batch):
            images=[];classes=[]
            for j in range(first,min(first+batch,n)):
                # indices far beyond 0..train_samples to avoid exact training-image duplicates
                im,cls=make_icon(1_000_000+j,size=size,seed=seed+10)
                images.append(torch.tensor(np.asarray(im,dtype=np.float32).copy()).permute(2,0,1)/127.5-1)
                classes.append(cls)
            x1=torch.stack(images).to(device); c=torch.tensor(classes,device=device)
            x0=torch.randn_like(x1);t=torch.rand(len(c),device=device)
            xt,target=flow_interpolant(x0,x1,t,sigma_min=float(saved.get('sigma_min',.01)))
            pred=model(xt,t,c)
            tot+=float(F.mse_loss(pred,target,reduction='sum').item())
            baseline+=float(target.square().sum().item())
    denom=n*3*size*size
    return {'held_out_conditional_mse':tot/denom,'zero_velocity_mse':baseline/denom,
            'held_out_n':n,'training_sample_overlap':'No index overlap; same procedural generator family',
            'caveat':'Not likelihood, FID, recognition score, or a paper benchmark'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--n',type=int,default=80);p.add_argument('--seed',type=int,default=454)
    p.add_argument('--out',default='flow_eval.json');a=p.parse_args()
    if a.n<1:raise ValueError('n must be positive')
    result=evaluate(a.checkpoint,n=a.n,seed=a.seed)
    Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
