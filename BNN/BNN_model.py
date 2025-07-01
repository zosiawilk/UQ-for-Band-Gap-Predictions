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

class BayesianNN(PyroModule):
    def __init__(self, input_dim, out_dim=1, hid_dim=64, n_hid_layers=3, prior_scale=0.1):
        super().__init__()

        self.activation = nn.Softplus()  # Better for regression and smoother than Tanh or ReLU

        # Dynamically build the layer sizes
        layer_sizes = [input_dim] + [hid_dim] * n_hid_layers + [out_dim]
        self.layers = PyroModuleList([])

        for i in range(len(layer_sizes) - 1):
            linear = PyroModule[nn.Linear](layer_sizes[i], layer_sizes[i + 1])
            # Xavier-inspired scaling for prior std dev
            weight_prior = dist.Normal(0., prior_scale * np.sqrt(2 / (layer_sizes[i] + layer_sizes[i + 1])))
            bias_prior = dist.Normal(0., prior_scale)

            linear.weight = PyroSample(weight_prior.expand([layer_sizes[i + 1], layer_sizes[i]]).to_event(2))
            linear.bias = PyroSample(bias_prior.expand([layer_sizes[i + 1]]).to_event(1))

            self.layers.append(linear)

        # Learnable global observation noise (log σ for numerical stability)
        self.log_sigma = pyro.param("log_sigma", torch.tensor(-1.0))  # log(σ) ≈ log(0.37)

    def forward(self, x, y=None):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))  # Hidden layers
        mu = self.layers[-1](x).squeeze(-1)  # Output layer

        sigma = torch.exp(self.log_sigma)  # Ensure positivity of σ

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

        return mu