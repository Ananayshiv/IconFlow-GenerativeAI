import torch
from iconflow.train import VelocityUNet,Icons,conditional_flow_loss,sample

def test_flow_training_and_sampling():
    model=VelocityUNet(base=8)
    ds=Icons(size=16,count=6)
    x=torch.stack([ds[i][0] for i in range(2)])
    c=torch.tensor([ds[i][1] for i in range(2)])
    loss=conditional_flow_loss(model,x,c);loss.backward()
    assert torch.isfinite(loss) and model.inconv.weight.grad is not None
    generated=sample(model,labels=c,size=16,steps=2)
    assert generated.shape==(2,3,16,16) and torch.isfinite(generated).all()


def test_editable_vector_paths(tmp_path):
    from iconflow.train import make_icon
    from iconflow.vectorize import raster_to_svg
    import xml.etree.ElementTree as ET
    img,_=make_icon(5,size=48)
    output=tmp_path/'icon.svg'
    report=raster_to_svg(img,output,colors=3,foreground='#4763D8')
    assert report['paths']>0
    root=ET.parse(output).getroot()
    assert root.tag.endswith('svg')
    assert any(e.tag.endswith('path') for e in root.iter())
    assert 'data:image' not in output.read_text()
