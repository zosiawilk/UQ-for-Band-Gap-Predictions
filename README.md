# Uncertainty Quantification for Band-Gap Prediction  

**Internship Project – “Uncertainty Quantification and Validation of Neural Network Models of Complex Physics”**

---

## 📑 Contents
- [Overview](#-overview)  
- [Workflow](#-workflow)  
- [Implemented Models](#-implemented-models)  
- [Quick Start](#-quick-start)  
- [Results so far](#-results-so-far)  
- [Road-map](#-road-map)  
- [Citation](#-citation)  

---

## 🔎 Overview
Predicting the electronic band gap (\(E_g\)) of crystalline materials is crucial for screening photovoltaics, thermoelectrics, and wide-band semiconductors.  

But a single deterministic number hides the real question:  

**➡How confident should we be in that prediction?**

This repository implements and benchmarks multiple **Uncertainty Quantification (UQ)** strategies on a ~100k entry subset of the **Materials Project** dataset.  


---

## Workflow

<p align="center">
  <img src="docs/workflow.png" alt="Workflow pipeline" width="800">
</p>

**Pipeline stages:**
1. **Data Collection & Featurisation** → Raw Materials Project data, matminer descriptors.  
2. **Feature Selection** → Filtering & dimensionality reduction (GBFSW, manual).  
3. **Model Training** → Train GPR, Bayesian NNs, MC-Dropout.  
4. **Uncertainty Quantification** → Estimate epistemic & aleatoric uncertainties.  
5. **Validation** → Coverage metrics, calibration curves, MAE vs experiment.

## Models Implemented

| Family                 | Implementation                           | UQ Output                   | Status |
|-------------------------|------------------------------------------|------------------------------|--------|
| **Gaussian Process**   | ARD kernel + feature selection           | closed-form μ, σ             | ✅ working |
| **Bayesian NN (MCMC)** | HMC/NUTS inference                       | posterior predictive μ, σ     | ✅ working |
| **Bayesian NN (VI)**   | Variational Inference (Bayes-by-Backprop)| approximate posterior μ, σ    | ✅ working |
| **MC-Dropout**         | Self-normalizing MLP + test-time dropout | epistemic + aleatoric σ      | ✅ working |
| **Direct-Error Model** | XGBoost on residuals                     | did not converge / not usable | ❌ failed |


## Results so far

| Model        | \(R^2\) | MSE   | NLL   |
|--------------|---------|-------|-------|
| BNN (MCMC)   | 0.844   | 0.312 | 0.483 |
| BNN (VI)     | 0.796   | 0.447 | 2.126 |
| MC Dropout   | 0.784   | 0.481 | 2.804 |
| GPR          | 0.757   | 0.535 | 1.234 |

