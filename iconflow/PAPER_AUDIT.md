# IconFlow — Paper fidelity and application extension

**Paper:** Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023, https://arxiv.org/abs/2210.02747.

**Faithful mathematical subset:** Conditioned Gaussian OT probability path with `x_t=(1-(1-sigma_min)t)x_0+t*x_1`, conditional target velocity `u_t=x_1-(1-sigma_min)x_0`, standard vector-field MSE, continuous time sampling, and integration of the learned ODE from Gaussian prior to data using explicit Euler. `test_paper_fidelity.py` verifies conditional path endpoints and velocity, nonzero gradients, numerical solver behavior, zero-field controls and image shape.

**Not a literal full-paper experimental replication:** A compact class-conditional pixel-space U-Net and synthetic 24-pixel icons replace the paper's larger benchmark models and image datasets. This is a correct *specific instance* of the Flow Matching framework, not reproduction of the ICLR paper's tables. Class conditioning is a downstream application choice.

**Independent extension:** `--solver heun` for second-order predictor/corrector sampling; `--export-svg` traces editable vector paths from generated raster icons. SVG contour extraction uses classical OpenCV and is not neural vector generation or a novel Flow Matching loss. Euler without SVG is the baseline.

**Validation status:** Run math and CLI tests, then full T4 training. On a held-out procedural icon sample, `evaluate_flow.py` measures conditional velocity prediction versus a trivial zero vector field. Such MSE does NOT measure perceptual fidelity; sample grids and human inspection still required. Large-dataset FID / source benchmark parity not achieved.

## V6 conditional extension (separate from paper method)

The additional `num_classes` embedding row is the *unconditional null token*. Training with `--cfg-dropout p` randomly substitutes the null label in `p` of training images while preserving the same Gaussian OT path and velocity objective. At sampling, `v_guided=v_uncond+s*(v_cond-v_uncond)`. `s=1` is the baseline conditional vector field; `s>1` requires **an appropriately dropout-trained checkpoint**. Unit tests check the velocity algebra, null-token gradient after the output head begins training, invalid guidance and the unconditional training path. This does not convert the paper baseline into an unconditional or pretrained model.
