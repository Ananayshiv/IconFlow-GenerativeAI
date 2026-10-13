import torch
import pytest
from iconflow.train import VelocityUNet,conditional_flow_loss,sample

class ToyVelocity(torch.nn.Module):
    num_classes=5
    def forward(self,x,t,c):
        # The classifier-free null token yields zero velocity;
        # all other labels yield 1: CFG with scale=2 must double it.
        return (c!=self.num_classes).float()[:,None,None,None].expand_as(x)

def test_classifier_free_guidance_math():
    m=ToyVelocity();labels=torch.tensor([0]);noise=torch.zeros(1,3,8,8)
    baseline=sample(m,labels,size=8,steps=3,noise=noise,clip_output=False,guidance_scale=1)
    double=sample(m,labels,size=8,steps=3,noise=noise,clip_output=False,guidance_scale=2)
    assert torch.allclose(baseline,torch.ones_like(baseline),atol=1e-6)
    assert torch.allclose(double,torch.full_like(double,2),atol=1e-6)

def test_cfg_null_dropout_trains_unconditional_embeddings():
    torch.manual_seed(11)
    net=VelocityUNet(base=8)
    x=torch.randn(2,3,16,16);labels=torch.tensor([0,1],dtype=torch.long)
    loss=conditional_flow_loss(net,x,labels,cfg_dropout=1.)
    # Output head is intentionally initialized to zero, so embeddings get no
    # gradient on step one. After one optimizer step, CFG null-token gradients flow.
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    loss.backward();assert torch.isfinite(loss)
    opt.step();opt.zero_grad()
    loss=conditional_flow_loss(net,x,labels,cfg_dropout=1.)
    loss.backward()
    assert net.label.weight.grad[-1].abs().sum()>0
    assert net.label.weight.grad[:-1].abs().sum()==0

def test_invalid_cfg_rejected():
    net=VelocityUNet(base=8)
    with pytest.raises(ValueError):
        conditional_flow_loss(net,torch.zeros(1,3,16,16),torch.zeros(1,dtype=torch.long),cfg_dropout=1.1)
    with pytest.raises(ValueError):
        sample(net,torch.tensor([1]),size=16,steps=1,guidance_scale=-1)
