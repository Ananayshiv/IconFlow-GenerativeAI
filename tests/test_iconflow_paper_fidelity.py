"""Mathematical checks of conditional OT path, velocity, and ODE solvers."""
import math
import torch
from torch import nn
from iconflow.train import flow_interpolant, sample, VelocityUNet, conditional_flow_loss


def test_gaussian_conditional_path_endpoints_velocity():
    a=torch.zeros(3,3,8,8);b=torch.ones_like(a)
    a[1]=2;a[2]=-.5
    t=torch.tensor([0.,.5,1.])
    sigma=.01
    xt,v=flow_interpolant(a,b,t,sigma)
    assert torch.allclose(xt[0],a[0])
    assert torch.allclose(xt[2],b[2]+sigma*a[2])
    assert torch.allclose(v,b-(1-sigma)*a)
    try:flow_interpolant(a,b,t,1.01)
    except ValueError:pass
    else:raise AssertionError('invalid sigma_min accepted')


def test_flow_loss_has_finite_gradients():
    model=VelocityUNet(base=8)
    x=torch.randn(2,3,16,16)
    y=torch.tensor([0,4])
    loss=conditional_flow_loss(model,x,y,sigma_min=.01,
                               noise=torch.zeros_like(x),times=torch.tensor([0.,1.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.inconv.weight.grad is not None


class LinearVelocity(nn.Module):
    def forward(self,x,t,c):
        return -x


def test_heun_converges_better_than_euler_for_linear_ode():
    model=LinearVelocity();labels=torch.tensor([1]);noise=torch.full((1,3,8,8),.5)
    true=.5/math.e
    e=sample(model,labels,size=8,steps=4,noise=noise,solver='euler',clip_output=False)
    h=sample(model,labels,size=8,steps=4,noise=noise,solver='heun',clip_output=False)
    assert abs(h.mean().item()-true)<abs(e.mean().item()-true)


def test_sampler_accepts_cpu_labels_and_validates_noise_shape():
    class Zeros(nn.Module):
        def forward(self,x,t,c): return torch.zeros_like(x)
    labels=torch.tensor([0,2]); noise=torch.randn(2,3,8,8)
    out=sample(Zeros(),labels,size=8,steps=5,solver='euler',noise=noise,clip_output=False)
    assert torch.allclose(out,noise)
    try: sample(Zeros(),labels,size=8,steps=1,noise=noise[:1])
    except ValueError: pass
    else: raise AssertionError('Should reject mismatched noise shapes')


def test_exact_constant_velocity_integration_euler_heun():
    class Constant(nn.Module):
        def forward(self,x,t,c): return torch.ones_like(x)*2
    labels=torch.tensor([0]); noise=torch.zeros(1,3,8,8)
    for solver in ('euler','heun'):
        out=sample(Constant(),labels,size=8,steps=3,solver=solver,noise=noise,clip_output=False)
        assert torch.allclose(out,torch.full_like(noise,2.0),atol=1e-6)
