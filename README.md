# Uncertainty Quantification for Band-Gap Prediction  
*Internship project – “Uncertainty Quantification and Validation of Neural
Network Models of Complex Physics”*

---

## Contents
| section | short description |
|---------|-------------------|
| [Overview](#overview) | What problem we solve and why UQ matters |
| [Workflow](#workflow) | 6-stage pipeline from raw MP data to calibrated uncertainties |
| [Implemented models](#implemented-models) | Direct-Error, BNN, MC-Dropout, GPR, Deep-Ensemble |
| [Quick start](#quick-start) | Conda install & one-liner training |
| [Results-so-far](#results-so-far) | snapshot metrics & example plots |
| [Road-map](#road-map) | next experiments from the presentation |
| [Citation](#citation) | how to reference this repo |

---

## Overview
Predicting the electronic **band gap (E<sub>g</sub>)** of crystalline materials
is crucial for screening photovoltaics, thermoelectrics and wide-band semiconductors.
A single deterministic number, however, hides the real question:

> *How confident should we be in that prediction?*

This repo implements and benchmarks multiple **Uncertainty Quantification (UQ)
strategies** on a 10 k-entry subset of the Materials Project.  
The code is the live counterpart to the internship presentation “Uncertainty
Quantification and Validation of Neural Network Models of Complex Physics”. :contentReference[oaicite:4]{index=4}

---

## Workflow
Materials Project CSV → Matminer features → GBFSW / manual
│ │ │
▼ ▼ ▼
data_collection featurization feature_selection
│
▼
┌──────────────────────────┐
│ model_training.py │ ← BNN / GPR / Dropout / Ensemble
└──────────────────────────┘
│
▼
uncertainty_estimation.py
│
▼
validation & plotting

yaml
Copy
Edit
Each stage is a standalone script; run `python stage_name.py -h` for options.

---

## Implemented models
| family | concrete implementation | UQ output |
|--------|-------------------------|-----------|
| **Direct Error Modelling** | XGB regressor on |ε| | point estimate of |ΔE<sub>g</sub>| |
| **Bayesian NN** | MCMC **and** Variational-Inference BNN | predictive μ, σ |
| **MC-Dropout** | Self-normalising MLP + Monte-Carlo passes | epistemic + aleatoric σ |
| **Gaussian Process** | Dynamic ARD kernel, SelectKBest | closed-form μ, σ |
| **Deep Ensemble** | 5 random-seed MLPs | ensemble variance |

Uncertainty is validated via **coverage** of 95 %/68 % credible intervals and
**MAE** against experimental band-gaps. :contentReference[oaicite:5]{index=5}

---

## Quick start
```bash
git clone https://github.com/<your-handle>/bandgap-uq.git
cd bandgap-uq

# create env (CPU TF by default)
conda env create -f environment.yml
conda activate bandgap-uq

# one-line MC-Dropout training & test-set evaluation
python src/train_bnn.py --model dropout --csv data/materials_data_10k_cleaned.csv
