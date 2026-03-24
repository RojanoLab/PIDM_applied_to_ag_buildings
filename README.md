# Physics-informed Diffusion Model for Turbulent Velocity Field Reconstruction

[![GitHub](https://img.shields.io/badge/GitHub-RojanoLab%2FPIDM-blue?logo=github)](https://github.com/RojanoLab/PIDM)

PyTorch implementation of

**A Physics-informed Diffusion Model for High-fidelity Flow Field Reconstruction**

Applied to turbulent airflow reconstruction around **agricultural buildings**.

> **Repository:** [https://github.com/RojanoLab/PIDM](https://github.com/RojanoLab/PIDM)

(Links to paper: <a href="https://www.sciencedirect.com/science/article/pii/S0021999123000670">Journal of Computational Physics</a> | <a href="https://arxiv.org/abs/2211.14680">arXiv</a>)

## Overview

Denoising Diffusion Probabilistic Models (DDPM) are used here to reconstruct high-fidelity 2D turbulent velocity fields (x- and y-components) around agricultural buildings from sparse or low-fidelity references. The model is trained exclusively on high-resolution velocity data and uses a **physics-informed conditioning signal** derived from the k-ε turbulence model to guide the reverse diffusion process. This conditioning enforces RANS residuals (continuity, momentum, and turbulent transport equations) during sampling, making reconstructions physically consistent without requiring paired low/high-resolution training data.

## Project Structure

```
PIDM/
├── main.py                                      # Entry point for guided sampling / reconstruction
├── configs/
│   └── kmflow_re1000_rs256_conditional.yml      # Sampling configuration
├── train_ddpm/
│   ├── main.py                                  # Entry point for model training
│   ├── train.sh                                 # Training shell script
│   └── configs/
│       └── vel_256_512_conditional.yml          # Training configuration
├── runners/
│   └── rs256_guided_diffusion2.py               # Guided diffusion sampler with k-ε conditioning
├── models/
│   ├── diffusion_new.py                         # UNet (unconditional and conditional variants)
│   └── ema.py                                   # Exponential Moving Average helper
├── functions/
│   ├── denoising_step.py                        # DDPM / DDIM denoising steps
│   └── process_data.py                          # Data loading and pre-processing
├── data/
│   ├── Vel_X.npy                                # X-velocity training data  (7, 144, 256, 512) uint8
│   ├── Vel_Y.npy                                # Y-velocity training data  (7, 144, 256, 512) uint8
│   ├── 256-512_Vel_X_stats.npz                  # Normalisation stats for Vel_X (mean, scale)
│   ├── 256-512_Vel_Y_stats.npz                  # Normalisation stats for Vel_Y (mean, scale)
│   └── prediction_k_eps_mean.npz               # k-ε turbulence model predictions (k, epsilon)
├── pretrained_weights/                          # Saved model checkpoints (.pth)
└── experiments/                                 # Output directory for reconstructions
```

## Dataset

The training data consists of 2D turbulent velocity fields with the following properties:

| File | Shape | dtype | Description |
|------|-------|-------|-------------|
| `Vel_X.npy` | `(7, 144, 256, 512)` | uint8 | X-component of velocity (7 sequences × 144 time steps × 256 × 512 px) |
| `Vel_Y.npy` | `(7, 144, 256, 512)` | uint8 | Y-component of velocity |
| `256-512_Vel_X_stats.npz` | scalars | float64 | `mean` and `scale` for Vel_X normalisation |
| `256-512_Vel_Y_stats.npz` | scalars | float64 | `mean` and `scale` for Vel_Y normalisation |
| `prediction_k_eps_mean.npz` | arrays | float32 | Turbulent kinetic energy `k` and dissipation rate `epsilon` from a RANS k-ε model |

Place all data files inside the `./data/` subdirectory before running any experiment.

For reconstruction (sampling), three additional reference files are expected at the root:

| File | Description |
|------|-------------|
| `ref1.npy` | Ground-truth x-velocity reference (sequence 8) |
| `ref2.1.npy` | Guided image conditioning signal g(u) |
| `seq_8_y_npy.npy` | Ground-truth y-velocity reference (sequence 8) |

## Environment

```
python 3.8
PyTorch 1.7 + CUDA 10.1 + torchvision 0.8.2
TensorBoard 2.11
Numpy 1.22
tqdm 4.59
einops 0.4.1
matplotlib 3.6.2
```

## Running the Experiments

### Step 1 — Model Training

From inside the `./train_ddpm/` subdirectory, run:

```bash
bash train.sh
```

or directly:

```bash
python main.py \
    --config ./vel_256_512_conditional.yml \
    --exp ./experiments/results/ \
    --doc ./weights/trained_UNet_nn_/ \
    --ni
```

Key training hyperparameters (set in `train_ddpm/configs/vel_256_512_conditional.yml`):

| Parameter | Value |
|-----------|-------|
| Image size | 512 × 256 |
| Channels | 3 |
| Batch size | 12 |
| Epochs | 1 000 |
| Iterations | 200 000 |
| Optimizer | Adam (lr = 2e-4) |
| Diffusion timesteps | 1 000 |
| Snapshot frequency | every 20 000 iterations |

Checkpoints are saved to:

```
train_ddpm/experiments/results/logs/weights/trained_UNet_nn_/
```

You can change the output location via the `--exp` and `--doc` arguments.

### Step 2 — Physics-informed Reconstruction (Sampling)

Place the trained checkpoint (e.g., `ckpt_Vel_X.pth`) into `./pretrained_weights/` and set `ckpt_path` accordingly in `configs/kmflow_re1000_rs256_conditional.yml`.

From the **root** directory of the repository, run:

```bash
python main.py \
    --config kmflow_re1000_rs256_conditional.yml \
    --seed 1234 \
    --sample_step 1 \
    --t 1000 \
    --r 20
```

Key sampling arguments:

| Argument | Description |
|----------|-------------|
| `--t` | Forward diffusion noise scale (number of timesteps to corrupt the reference) |
| `--r` / `--reverse_steps` | Number of reverse (denoising) steps |
| `--seed` | Random seed for reproducibility |
| `--sample_step` | Number of sampling repetitions |

The `guidance_weight` in the config controls the strength of the k-ε physics residual signal during sampling. Results are saved under `./experiments/` in a subfolder named after the run parameters (e.g., `guided_recons__t1000_r20_w3.0/`).

## References

If you find this repository useful for your research, please cite the following work.

```bibtex
@article{shu2023physics,
  title={A Physics-informed Diffusion Model for High-fidelity Flow Field Reconstruction},
  author={Shu, Dule and Li, Zijie and Farimani, Amir Barati},
  journal={Journal of Computational Physics},
  pages={111972},
  year={2023},
  publisher={Elsevier}
}
```

This implementation is based on / inspired by:

- [https://github.com/ermongroup/SDEdit](https://github.com/ermongroup/SDEdit) (SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations)
- [https://github.com/ermongroup/ddim](https://github.com/ermongroup/ddim) (Denoising Diffusion Implicit Models)
