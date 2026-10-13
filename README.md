# IconFlow — Conditional Flow Matching for Procedural Icons

From-scratch PyTorch velocity-field regression using conditional **Flow Matching** with a class-conditional neural network. Optional classifier-free guidance (CFG), Euler and Heun ODE integration, and contour-to-SVG export are implemented as independent extensions.

## Measured results — held-out procedural data
| Model | Conditional velocity MSE ↓ | Zero-velocity MSE |
|---|---:|---:|
| Baseline (3,000 steps, 24×24) | 0.07101 | 1.68708 |
| CFG-trained (3,000 steps, 24×24) | **0.06802** | 1.68708 |

The CFG-trained model's velocity MSE is approximately **4.2% lower**, but this does not establish generation-quality improvement due to inference-time guidance. User-reviewed grids show recognizable stars, circles, lightning bolts, chat bubbles, and diamonds, with visible blur and pixelation. No FID, official benchmark, or general-purpose vector-generation claim is made.

## Method
A Gaussian conditional path `x_t = (1-(1-σ_min)t)x_0 + t x_1` defines the analytic training target `u_t = x_1 - (1-σ_min)x_0`. The learned marginal velocity field regresses the conditional-mean target. Euler and Heun samplers integrate the field; high-order sampling is compared at matched function-evaluation cost.

## Reproduce
```bash
pip install -r requirements.txt
python -m iconflow.train --steps 3000 --batch-size 32 --size 24 --width 32 --samples 3000 --seed 42 --cfg-dropout 0 --out runs/baseline
python -m iconflow.train --steps 3000 --batch-size 32 --size 24 --width 32 --samples 3000 --seed 42 --cfg-dropout 0.15 --out runs/cfg
python -m iconflow.evaluate_flow --checkpoint runs/cfg/flow_model.pt --n 120 --out runs/cfg/held_out_flow.json
python -m pytest -q
```

The data are synthetic icons and the training runs are small-scale. Model checkpoints and samples remain in Google Drive pending a dedicated artifacts release.

**Paper:** Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
