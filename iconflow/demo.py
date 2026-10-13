"""Class-conditioned creative asset generator + explicit classical SVG trace postprocess."""
import argparse
import tempfile
from pathlib import Path
import torch
from PIL import Image
from iconflow.train import VelocityUNet,CLASSES,sample,to_pil
from iconflow.vectorize import raster_to_svg

PALETTES={'Original (sampled)':None,'Cobalt':'#4763D8','Coral':'#E95A67','Jade':'#269D87','Ink':'#152642'}


def generate_one(model,device,size,kind,seed,steps,palette,solver="euler",guidance=1.):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type=='cuda' else []):
        torch.manual_seed(int(seed))
        if device.type=='cuda':torch.cuda.manual_seed_all(int(seed))
        labels=torch.tensor([CLASSES.index(kind)],device=device)
        img=to_pil(sample(model,labels,size=size,steps=int(steps),device=device,solver=solver,guidance_scale=float(guidance))[0])
    dest=Path(tempfile.mkdtemp(prefix='icon_assets_'))
    img.resize((384,384),Image.Resampling.NEAREST).save(dest/'icon.png')
    report=raster_to_svg(img,dest/'editable_icon.svg',foreground=PALETTES[palette])
    return Image.open(dest/'icon.png'),str(dest/'editable_icon.svg'),f'{report["paths"]} vector contours (heuristic; inspect and refine in Illustrator)'


def create_app(checkpoint='outputs/flow_model.pt'):
    import gradio as gr
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=VelocityUNet(base=saved.get('width',32)).to(device).eval()
    model.load_state_dict(saved['model'])
    cfg_trained=bool(saved.get('cfg_dropout',0)>0)
    def predict(kind,seed,steps,palette,solver,guidance):
        if float(guidance)!=1. and not cfg_trained:
            raise gr.Error('Train with --cfg-dropout >0 before using classifier-free guidance')
        return generate_one(model,device,saved.get('size',24),kind,seed,steps,palette,solver,guidance)
    return gr.Interface(predict,
        inputs=[gr.Dropdown(CLASSES,value='star',label='Asset class'),
                gr.Slider(0,9999,step=1,value=42,label='Seed'),
                gr.Slider(10,100,step=5,value=35,label='ODE steps'),
                gr.Dropdown(list(PALETTES),value='Cobalt',label='Vector palette'),
                gr.Dropdown(['euler','heun'],value='heun',label='ODE sampler'),
                gr.Slider(0,3,step=.25,value=1,label='Classifier-free guidance scale (needs CFG-trained checkpoint)')],
        outputs=[gr.Image(label='Generated raster'),gr.File(label='Editable SVG'),gr.Textbox(label='Trace diagnostics')],
        title='IconFlow · Generative Icons + Editable Vector Export',
        description='From-scratch conditional Flow Matching + optional CFG + ODE solver. SVG uses classical contour tracing. A checkpoint trained without label dropout only supports scale=1. Quality depends on training.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='outputs/flow_model.pt')
    create_app(p.parse_args().checkpoint).launch(share=False)
