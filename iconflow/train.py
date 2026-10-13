"""Conditional flow matching from scratch: straight-line velocity field + Euler ODE sampler.
ICLR 2023 paper-inspired SMALL pixel-space model trained on procedural icons.
python train.py --steps 1000 --batch-size 32  (Colab T4 recommended)
"""
import argparse,math,csv,random
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

CLASSES=['circle','star','lightning','chat','diamond']
COLORS=['#4C63D2','#F05D66','#1FA89A','#F5A623','#9D52C7']


def make_icon(index,size=24,seed=43):
    r=random.Random(seed+index*7349);cls=index%len(CLASSES);ss=size*3
    bg=(r.randrange(235,255),r.randrange(235,255),r.randrange(235,255))
    im=Image.new('RGB',(ss,ss),bg);d=ImageDraw.Draw(im)
    color=r.choice(COLORS)
    cx=(.5+r.uniform(-.10,.10))*ss;cy=(.5+r.uniform(-.10,.10))*ss
    a=r.uniform(.24,.36)*ss;x0,y0,x1,y1=(cx-a,cy-a,cx+a,cy+a)
    if cls==0:d.ellipse((x0,y0,x1,y1),fill=color)
    elif cls==1:
        outer=[]
        for k in range(10):
            ang=-math.pi/2+k*math.pi/5; rad=a if k%2==0 else a*.44
            outer.append((cx+math.cos(ang)*rad,cy+math.sin(ang)*rad))
        d.polygon(outer,fill=color)
    elif cls==2:
        pts=[(cx+.05*a,cy-a),(cx-.8*a,cy+.15*a),(cx-.12*a,cy+.15*a),(cx-.3*a,cy+a),(cx+.75*a,cy-.24*a),(cx+.12*a,cy-.24*a)]
        d.polygon(pts,fill=color)
    elif cls==3:
        d.rounded_rectangle((x0,y0,x1,y1-.1*a),radius=.25*a,fill=color)
        d.polygon([(cx-.42*a,cy+.7*a),(cx-.42*a,cy+1.1*a),(cx+.03*a,cy+.55*a)],fill=color)
        for dx in (-.4,0,.4):d.ellipse((cx+dx*a-.08*a,cy-.08*a,cx+dx*a+.08*a,cy+.08*a),fill='white')
    else:d.polygon([(cx,cy-a),(cx+a,cy),(cx,cy+a),(cx-a,cy)],fill=color)
    return im.resize((size,size),resample=Image.Resampling.LANCZOS),cls


class Icons(Dataset):
    def __init__(self,size=24,count=3000,seed=43):self.size=size;self.count=count;self.seed=seed
    def __len__(self):return self.count
    def __getitem__(self,idx):
        image,cls=make_icon(idx,self.size,self.seed)
        x=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()).permute(2,0,1)/127.5-1
        return x,cls


class TimeEmbed(nn.Module):
    def __init__(self,dim):
        super().__init__();self.dim=dim
        self.mlp=nn.Sequential(nn.Linear(dim,dim*4),nn.SiLU(),nn.Linear(dim*4,dim))
    def forward(self,t):
        half=self.dim//2
        f=torch.exp(-math.log(10000)*torch.arange(half,device=t.device,dtype=t.dtype)/max(half-1,1))
        z=t[:,None]*f[None,:]*1000
        embedding=torch.cat([z.sin(),z.cos()],dim=-1)
        return self.mlp(embedding)


class ResBlock(nn.Module):
    def __init__(self,inch,outch,edim):
        super().__init__()
        self.norm1=nn.GroupNorm(8,inch);self.conv1=nn.Conv2d(inch,outch,3,padding=1)
        self.norm2=nn.GroupNorm(8,outch);self.conv2=nn.Conv2d(outch,outch,3,padding=1)
        self.emb=nn.Linear(edim,outch);self.shortcut=nn.Conv2d(inch,outch,1) if inch!=outch else nn.Identity()
    def forward(self,x,e):
        z=self.conv1(F.silu(self.norm1(x)))
        z=z+self.emb(F.silu(e))[:,:,None,None]
        return self.shortcut(x)+self.conv2(F.silu(self.norm2(z)))


class VelocityUNet(nn.Module):
    """Tiny class-conditioned U-Net, built without pretrained diffusion/flow libraries."""
    def __init__(self,base=32,num_classes=5):
        super().__init__()
        ed=base*4
        self.time=TimeEmbed(ed);self.num_classes=num_classes;self.label=nn.Embedding(num_classes+1,ed)  # final index is CFG null token
        self.inconv=nn.Conv2d(3,base,3,padding=1)
        self.enc1=ResBlock(base,base,ed)
        self.down1=nn.Conv2d(base,base*2,3,stride=2,padding=1)
        self.enc2=ResBlock(base*2,base*2,ed)
        self.down2=nn.Conv2d(base*2,base*4,3,stride=2,padding=1)
        self.mid=ResBlock(base*4,base*4,ed)
        self.dec2=ResBlock(base*6,base*2,ed)
        self.dec1=ResBlock(base*3,base,ed)
        self.outnorm=nn.GroupNorm(8,base);self.outconv=nn.Conv2d(base,3,3,padding=1)
        nn.init.zeros_(self.outconv.weight);nn.init.zeros_(self.outconv.bias)
    def forward(self,x,t,c):
        if c.dtype != torch.long or bool(((c<0)|(c>self.num_classes)).any()):
            raise ValueError('Labels must be LongTensor in [0, num_classes], with num_classes as null token')
        e=self.time(t)+self.label(c)
        x1=self.enc1(self.inconv(x),e)
        x2=self.enc2(self.down1(x1),e)
        z=self.mid(self.down2(x2),e)
        z=F.interpolate(z,size=x2.shape[-2:],mode='nearest')
        z=self.dec2(torch.cat([z,x2],dim=1),e)
        z=F.interpolate(z,size=x1.shape[-2:],mode='nearest')
        z=self.dec1(torch.cat([z,x1],dim=1),e)
        return self.outconv(F.silu(self.outnorm(z)))


def flow_interpolant(x0,x1,t,sigma_min=.01):
    """Gaussian OT conditional path and velocity (Lipman et al., ICLR 2023).

    x_t=(1-(1-sigma_min)t)*x0+t*x1
    u_t=x1-(1-sigma_min)*x0. sigma_min>0 avoids a degenerate final
    conditional distribution. sigma_min=0 is the straight-line limit.
    """
    if not (0<=sigma_min<1): raise ValueError('sigma_min must be in [0,1)')
    t=t.reshape(-1,1,1,1)
    xt=(1-(1-sigma_min)*t)*x0+t*x1
    target=x1-(1-sigma_min)*x0
    return xt,target


def conditional_flow_loss(model,x1,labels,sigma_min=.01,noise=None,times=None,cfg_dropout=0.):
    """FM objective. With cfg_dropout>0 this learns an unconditional null-token branch.

    CFG is an independent extension, not a requirement of the Lipman et al. objective.
    """
    if not 0. <= cfg_dropout <= 1.:
        raise ValueError('cfg_dropout must lie in [0,1]')
    x0=torch.randn_like(x1) if noise is None else noise
    t=torch.rand(len(x1),device=x1.device) if times is None else times
    xt,velocity=flow_interpolant(x0,x1,t,sigma_min)
    if cfg_dropout:
        mask=torch.rand(len(labels),device=labels.device)<cfg_dropout
        labels=torch.where(mask,torch.full_like(labels,model.num_classes),labels)
    prediction=model(xt,t,labels)
    return F.mse_loss(prediction,velocity)


@torch.no_grad()
def sample(model,labels,size=24,steps=30,device='cpu',solver='euler',
           noise=None,clip_output=True,guidance_scale=1.):
    """ODE dx/dt=v_theta(x,t); paper-faithful Euler plus Heun EXTENSION.

    Euler evaluates velocity at LEFT time i/steps, not (i+.5)/steps.
    Heun (predictor-corrector) uses an additional function evaluation.
    Clipping is only for final image rendering, never during ODE steps.
    """
    if solver not in ('euler','heun'): raise ValueError('solver must be euler/heun')
    if steps<1:raise ValueError('steps must be positive')
    if guidance_scale < 0:raise ValueError('guidance_scale must be nonnegative')
    model.eval()
    labels=labels.to(device=device,dtype=torch.long)
    if noise is not None and tuple(noise.shape)!=(len(labels),3,size,size):
        raise ValueError('Noise must have shape (len(labels),3,size,size)')
    x=(torch.randn((len(labels),3,size,size),device=device) if noise is None
       else noise.to(device).clone())
    dt=1/steps
    def guided_velocity(x,t):
        conditioned=model(x,t,labels)
        if guidance_scale==1.:
            return conditioned
        unconditioned=model(x,t,torch.full_like(labels,model.num_classes))
        return unconditioned+guidance_scale*(conditioned-unconditioned)
    for i in range(steps):
        t=torch.full((len(labels),),i/steps,device=device)
        v=guided_velocity(x,t)
        if solver=='heun':
            next_t=torch.full((len(labels),),(i+1)/steps,device=device)
            v_next=guided_velocity(x+dt*v,next_t)
            x=x+.5*dt*(v+v_next)
        else:
            x=x+dt*v
    return x.clamp(-1,1) if clip_output else x

def to_pil(x):
    a=((x.detach().cpu().permute(1,2,0).clamp(-1,1)+1)*127.5).numpy().astype('uint8')
    return Image.fromarray(a)


def save_grid(images,path,size=24,cols=5,scale=6):
    count=images.shape[0];rows=math.ceil(count/cols)
    grid=Image.new('RGB',(cols*size*scale,rows*size*scale),'white')
    for n,arr in enumerate(images):
        img=to_pil(arr).resize((size*scale,size*scale),Image.Resampling.NEAREST)
        grid.paste(img,((n%cols)*size*scale,(n//cols)*size*scale))
    grid.save(path)


def run(args):
    torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.size%4!=0:raise ValueError('--size must be divisible by 4')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    saved=torch.load(args.checkpoint,map_location=device,weights_only=True) if args.checkpoint else None
    effective_width=saved.get('width',args.width) if isinstance(saved,dict) else args.width
    effective_size=saved.get('size',args.size) if isinstance(saved,dict) else args.size
    if args.sample_only and args.guidance_scale != 1 and (not isinstance(saved,dict) or saved.get('cfg_dropout',0.)<=0):
        raise ValueError('Classifier-free guidance requires checkpoint trained with --cfg-dropout > 0')
    model=VelocityUNet(base=effective_width).to(device)
    if saved is not None:
        model.load_state_dict(saved['model'] if 'model' in saved else saved)
    labels=torch.tensor([i%5 for i in range(20)],device=device)
    if args.sample_only:
        save_grid(sample(model,labels,size=effective_size,steps=args.sample_steps,device=device,solver=args.solver,guidance_scale=args.guidance_scale),out/'generated_icons.png',effective_size)
        print('Wrote',out/'generated_icons.png')
        if args.export_svg:
            from iconflow.vectorize import vectorize_grid
            info=vectorize_grid(Image.open(out/'generated_icons.png'),out/'editable_svg_icons',effective_size)
            print('Exported',len(info),'editable SVG candidates')
        return
    ds=Icons(size=args.size,count=args.samples,seed=args.seed)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    itr=iter(dl);history=[]
    for step in range(1,args.steps+1):
        try:x,c=next(itr)
        except StopIteration:itr=iter(dl);x,c=next(itr)
        x=x.to(device);c=c.to(device);model.train()
        loss=conditional_flow_loss(model,x,c,sigma_min=args.sigma_min,cfg_dropout=args.cfg_dropout)
        optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
        history.append([step,float(loss.detach())])
        if step==1 or step%max(1,args.steps//10)==0:
            print(f'step {step}/{args.steps} flow loss {float(loss.detach()):.4f}')
    torch.save({'model':model.state_dict(),'size':args.size,'width':args.width,'steps':args.steps,'sigma_min':args.sigma_min,'cfg_dropout':args.cfg_dropout},out/'flow_model.pt')
    with (out/'training.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['step','flow_mse']);w.writerows(history)
    # Same evaluation seeds across runs, generate one independent image per category.
    torch.manual_seed(args.seed+999)
    if args.guidance_scale!=1 and args.cfg_dropout<=0:
        raise ValueError('--guidance-scale !=1 requires --cfg-dropout > 0 at training')
    generated=sample(model,labels,size=args.size,steps=args.sample_steps,device=device,solver=args.solver,guidance_scale=args.guidance_scale)
    save_grid(generated,out/'generated_icons.png',args.size)
    if args.export_svg:
        from iconflow.vectorize import vectorize_grid
        info=vectorize_grid(Image.open(out/'generated_icons.png'),out/'editable_svg_icons',args.size)
        print('Exported',len(info),'editable SVG candidates')
    reference=torch.stack([ds[i][0] for i in range(20)])
    save_grid(reference,out/'training_examples.png',args.size)
    import json
    (out/'metrics.json').write_text(json.dumps({'loss_initial':history[0][1],'loss_final':history[-1][1],
        'loss_last_50_mean':float(np.mean([v for _,v in history[-50:]])),
        'classes':CLASSES,'cfg_dropout':args.cfg_dropout,'guidance_scale':args.guidance_scale,'solver':args.solver,'evaluation_note':'Flow matching MSE is not an image-quality metric; visually inspect generated grid and extend with FID on a held-out image dataset.'},indent=2))
    print('Saved flow_model.pt, generated_icons.png, training_examples.png, training.csv, metrics.json to',out,'device',device)


def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--samples',type=int,default=3000)
    p.add_argument('--size',type=int,default=24);p.add_argument('--width',type=int,default=32)
    p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--sample-steps',type=int,default=30)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--out',default='outputs')
    p.add_argument('--checkpoint',default='');p.add_argument('--sample-only',action='store_true')
    p.add_argument('--sigma-min',type=float,default=.01,help='Gaussian OT conditional path endpoint variance; 0 is straight-line limit')
    p.add_argument('--solver',choices=['euler','heun'],default='euler',help='Heun is an optional sampling extension')
    p.add_argument('--export-svg',action='store_true',help='Contour-trace sampled images into editable SVGs (requires opencv)')
    p.add_argument('--cfg-dropout',type=float,default=0.0,help='EXTENSION: fraction of training labels replaced by unconditional null token (e.g. 0.15)')
    p.add_argument('--guidance-scale',type=float,default=1.0,help='EXTENSION: CFG generation scale; 1.0 reproduces base conditional velocity')
    return p.parse_args(argv)

if __name__=='__main__':run(parse_args())
