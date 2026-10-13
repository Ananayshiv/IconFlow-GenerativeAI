# Reported Colab run

- Conditional Flow Matching baseline and classifier-free-dropout variant, 3,000 steps each, seed 42.
- Samples generated at resolution 24×24.
- Held-out conditional velocity MSE: baseline **0.07101**, CFG-trained **0.06802**; zero-velocity baseline **1.68708** (120 evaluation samples).
- Output grids show recognizable procedural icons. Low MSE does **not** establish sample quality, and training with condition dropout does not by itself quantify classifier-free guidance benefits.

These are user-provided Colab results, not official Flow Matching paper benchmarks or independently reproduced GitHub metrics.
