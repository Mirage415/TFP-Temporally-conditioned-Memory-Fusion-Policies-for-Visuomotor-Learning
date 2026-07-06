# TFP: Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning

## Overview

This repository contains the research code for **TFP: Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning**.

TFP addresses a common limitation of reactive vision-language-action policies: in stage-dependent manipulation, the current observation alone may not determine the correct next action. Visually similar states can require different actions depending on latent task progress, previous contacts, occlusions, or completed subgoals.

TFP maintains an episode-local latent belief with Liquid Time-Constant dynamics and injects the updated belief directly into the flow-matching action decoder through AdaLN-style modulation. This allows temporally accumulated task context to shape generated action chunks, instead of serving only as passive history context.

## Results

| Benchmark / task | Reactive pi0.5 | TFP |
| --- | ---: | ---: |
| LIBERO average success | 96.85% | 98.75% |
| LIBERO Long-10 | 92.4% | 97.0% |
| LIBERO-plus average robustness | 91.4% | 93.77% |
| MIKASA-Robo ShellGameTouch | - | 75.0% |
| Galaxea A1 object swap | 3/20 | 15/20 |
| Galaxea A1 counting pick-place | 8/20 | 18/20 |

Mechanistic analyses show that LTC write-gain changes are about 6x larger near manipulation events than in far non-event phases. Hidden-state interventions further show that changing only the learned belief can change generated action chunks under fixed observation, robot state, language instruction, and flow-matching noise.

## Citation

```bibtex
@article{liang2026tfp,
  title   = {TFP: Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning},
  author  = {Liang, Yushen and Peng, Yue and Jin, Baosheng and Zhang, Tianluo and Zhang, Xinyu and Zhou, Shuyi and Chen, Zhuoran and Liu, Xinqi and Wan, Shenji},
  year    = {2026}
}
```
