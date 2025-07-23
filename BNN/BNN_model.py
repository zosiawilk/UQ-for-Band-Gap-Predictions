import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroModuleList
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""
    ======================== Bayesian Neural Network (BNN) model using Pyro. =========================

    - Class Definition:
      `BayesianNN` inherits from `PyroModule`, a Pyro-enhanced version of `nn.Module`,
      allowing neural network components to be treated probabilistically.

    - __init__ method:
      - `self.fc1`: First Bayesian linear layer with `input_dim` inputs and 100 outputs.
        - Weights and biases are random variables sampled from a normal distribution (`Normal(0, 0.1)`).
      - `self.fc2`: Second Bayesian linear layer with 100 inputs and 100 outputs.
        - Like `fc1`, its weights and biases follow a `Normal(0, 0.1)` distribution.
      - `self.out`: Final output Bayesian linear layer with 100 inputs and 1 output.
        - Also uses `Normal(0, 0.1)` priors for weights and biases.

    - forward method:
      - The input `x` flows through two hidden layers (`fc1` and `fc2`) with `softplus` activation functions.
      - The output layer `self.out` produces a predicted mean (`mean`) of the band gap.
      - A global observation noise `sigma` is sampled from a uniform distribution between 0 and 1.
      - Using `pyro.plate`, the target `y` is modeled as a normal distribution with the predicted mean and standard deviation `sigma`.
      - The function returns the predicted mean of the output distribution.

    Summary:
    This Bayesian neural network includes two hidden layers and models uncertainty in both the weights and predictions.
    By treating all weights and biases as distributions, it captures epistemic uncertainty (uncertainty due to limited data).
    The network outputs both predictions and a principled estimate of uncertainty, making it suitable for scientific
    applications like band gap prediction where confidence in results is critical.
"""

import math, torch, torch.nn as nn, pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class BayesianNN(PyroModule):
    def __init__(self, input_dim: int, hidden=(128, 64)):
        super().__init__()
        h1, h2 = hidden               # still tiny: (313*128)+(128*64)+… ≈ 25 k params

        # ----- layer 1 ---------------------------------------------------
        self.fc1 = PyroModule[nn.Linear](input_dim, h1)
        τ1 = 1 / math.sqrt(input_dim)
        self.fc1.weight = PyroSample(dist.Normal(0., τ1).expand([h1, input_dim]).to_event(2))
        self.fc1.bias   = PyroSample(dist.Normal(0., τ1).expand([h1]).to_event(1))

        # ----- layer 2 ---------------------------------------------------
        self.fc2 = PyroModule[nn.Linear](h1, h2)
        τ2 = 1 / math.sqrt(h1)
        self.fc2.weight = PyroSample(dist.Normal(0., τ2).expand([h2, h1]).to_event(2))
        self.fc2.bias   = PyroSample(dist.Normal(0., τ2).expand([h2]).to_event(1))

        # ----- output head ----------------------------------------------
        self.out = PyroModule[nn.Linear](h2, 1)
        τ3 = 1 / math.sqrt(h2)
        self.out.weight = PyroSample(dist.Normal(0., τ3).expand([1, h2]).to_event(2))
        self.out.bias   = PyroSample(dist.Normal(0., τ3).expand([1]).to_event(1))

    # -------------------------------------------------------------------
    def forward(self, x, y=None):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.out(x).squeeze(-1)                 # shape [N]

        sigma = pyro.sample("sigma", dist.HalfCauchy(1.))  # scalar noise scale

        with pyro.plate("data", x.size(0)):
            pyro.sample("obs", dist.Normal(mean, sigma), obs=y)

        return mean
