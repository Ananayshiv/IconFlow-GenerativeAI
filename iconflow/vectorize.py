"""Real editable SVG contours from pixel-space generated icon PNGs.

Quantize the source image and trace simplified contours. Output contains vector
<path> elements, no PNG/data URI. Requires opencv-python-headless and Pillow.
Not vector diffusion; this is a classical, separately attributed postprocess.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter


def raster_to_svg(image,output,colors=4,simplify=.012,blur=.3,foreground=None):
    import cv2
    im=image.convert('RGB')
    im=im.filter(ImageFilter.GaussianBlur(radius=max(0,float(blur))))
    rgb=np.asarray(im,dtype=np.uint8)
    h,w=rgb.shape[:2]
    # Median border color is the canvas background estimate.
    edge=np.concatenate([rgb[0,:,:],rgb[-1,:,:],rgb[:,0,:],rgb[:,-1,:]],axis=0)
    bg=np.median(edge,axis=0).astype(np.uint8)
    samples=rgb.reshape(-1,3).astype(np.float32)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,25,.25)
    cv2.setRNGSeed(1)
    _,idx,centers=cv2.kmeans(samples,max(2,int(colors)),None,criteria,4,cv2.KMEANS_PP_CENTERS)
    centers=np.clip(centers,0,255).astype(np.uint8)
    group=idx.reshape(h,w)
    # Majority border cluster designated background; remaining regions become paths.
    border_labels=np.concatenate([group[0,:],group[-1,:],group[:,0],group[:,-1]])
    bg_label=int(np.bincount(border_labels,minlength=len(centers)).argmax())
    fill_bg='#%02x%02x%02x'%tuple(int(v) for v in centers[bg_label])
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w*8}" height="{h*8}">',
           '<title>Editable vector traces from a generated raster icon</title>',
           f'<rect id="editable-background" width="{w}" height="{h}" fill="{fill_bg}"/>']
    paths=0
    for lab in range(len(centers)):
        if lab==bg_label:continue
        mask=(group==lab).astype(np.uint8)*255
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        color=(foreground if foreground else '#%02x%02x%02x'%tuple(int(v) for v in centers[lab]))
        for contour in contours:
            if cv2.contourArea(contour)<max(3,w*h*.003):continue
            approx=cv2.approxPolyDP(contour,max(.15,float(simplify)*cv2.arcLength(contour,True)),True).reshape(-1,2)
            if len(approx)<3:continue
            d='M '+' L '.join(f'{int(x)} {int(y)}' for x,y in approx)+' Z'
            lines.append(f'<path id="vector-shape-{paths+1}" d="{d}" fill="{color}"/>')
            paths+=1
    lines.append('</svg>')
    Path(output).write_text('\n'.join(lines),encoding='utf-8')
    return {'width':w,'height':h,'paths':paths,'background':fill_bg,'output':str(output)}


def vectorize_grid(image,output_dir,icon_size,cols=5,count=20,scale=6):
    """Slice nearest-neighbor tiled grid and SVG-trace each upscaled tile."""
    grid=image.convert('RGB');out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    info=[]
    for i in range(count):
        x=(i%cols)*icon_size*scale;y=(i//cols)*icon_size*scale
        tile=grid.crop((x,y,x+icon_size*scale,y+icon_size*scale))
        # Match original low-res structure before contour extraction.
        tile=tile.resize((icon_size,icon_size),Image.Resampling.LANCZOS)
        dest=out/f'icon_{i:02d}.svg'
        report=raster_to_svg(tile,dest)
        info.append(report)
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--out',default='editable_icon.svg')
    p.add_argument('--colors',type=int,default=4)
    a=p.parse_args()
    print(raster_to_svg(Image.open(a.image),a.out,colors=a.colors))
