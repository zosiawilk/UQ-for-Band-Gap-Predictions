import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
======================== Bayesian Neural Network (BNN) model using Pyro. =====================

- Class Definition:
  `BayesianNN` inherits from `PyroModule`, which is a Pyro wrapper for PyTorch modules,
    enabling probabilistic programming.

- __init__ method:
  - `self.fc1`: First fully connected (linear) layer with `input_dim` inputs and 50 outputs.  
    - Its weights and biases are not fixed, but are random variables sampled 
    from a standard normal distribution (`Normal(0, 1)`), making them Bayesian parameters.
  - `self.fc2`: Second linear layer with 50 inputs and 1 output.  
    - Its weights and biases are also Bayesian, sampled from `Normal(0, 1)`.

- forward method:  
  - Input `x` passes through `fc1` and a ReLU activation.
  - Output is passed through `fc2` to get the predicted mean (`mean`).
  - The observation noise `sigma` is sampled from a uniform distribution between 0 and 1.
  - For each data point (using `pyro.plate` for vectorization), the observed target `y` is modeled
    as a normal distribution with the predicted mean and noise `sigma`.
  - Returns the predicted mean.

Summary:  
This is a Bayesian neural network with one hidden layer. All weights and biases are treated as random variables,
so the model learns distributions over parameters, not just point estimates. 
This allows for uncertainty quantification in predictions.
"""

class BayesianNN(PyroModule):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](input_dim, 50)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([50, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([50]).to_event(1))
        
        self.fc2 = PyroModule[nn.Linear](50, 1)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([1, 50]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))

    def forward(self, x, y=None):
        x = torch.relu(self.fc1(x))
        mean = self.fc2(x).squeeze(-1)
        sigma = pyro.sample("sigma", dist.Uniform(0., 1.))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean
