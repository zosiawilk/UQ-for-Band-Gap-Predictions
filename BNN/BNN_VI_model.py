import pyro
import pyro.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from pyro.nn import PyroModule, PyroSample, PyroModuleList

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.nn import PyroModule, PyroSample, PyroModuleList

"""
This class defines a fully Bayesian neural network using Pyro. Suitable for regression tasks with uncertainty quantification.

---------- Architecture Overview:--------------------
=> Input: input_dim-dimensional feature vector.

=> Hidden Layers: n_hid_layers fully connected layers with hid_dim neurons and ReLU activation.

=> Output: Single scalar prediction (e.g., band gap value).

=> Weight/Bias Priors: All layer parameters are drawn from Normal distributions

 ---------------- Components -------------------------------------------------------------------------------------------:
-----------------Part ------------------------------|------------------Description------------------------------
PyroModuleList	                                    |    Holds the list of hidden layers (like nn.ModuleList but 
                                                    |    supports stochastic parameters).
PyroSample	                                        |    Wraps parameters as random variables with specified priors.
F.relu(layer(x))	                                |    Applies ReLU activation after each hidden layer.
pyro.plate("data", x.shape[0])	                    |    Vectorized handling of independent samples in a batch.
pyro.sample("obs", dist.Normal(mean, sigma), obs=y)	|    Defines the likelihood distribution for observed data.

----------- Model Behavior:-----------------------------
=> Returns the mean prediction of the output distribution.

=> Learns posterior distributions over parameters instead of point estimates.

=> During training, you use tools like SVI + AutoDiagonalNormal to approximate the posterior.

! Important Notes:
* sigma = 0.1 is fixed — you may consider learning this as well for heteroscedastic models.

* expand(...).to_event(...) ensures correct handling of parameter shapes and dependencies.
"""


class BNN(PyroModule):
    def __init__(self, input_dim, hid_dim=50, n_hid_layers=2, prior_scale=1.0):
        super().__init__()
        self.hidden_layers = PyroModuleList([])
        last_dim = input_dim
        for i in range(n_hid_layers):
            layer = PyroModule[nn.Linear](last_dim, hid_dim)
            layer.weight = PyroSample(dist.Normal(0., prior_scale)
                                      .expand([hid_dim, last_dim]).to_event(2))
            layer.bias = PyroSample(dist.Normal(0., prior_scale)
                                    .expand([hid_dim]).to_event(1))
            self.hidden_layers.append(layer)
            last_dim = hid_dim

        self.out = PyroModule[nn.Linear](last_dim, 1)
        self.out.weight = PyroSample(dist.Normal(0., prior_scale)
                                     .expand([1, last_dim]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., prior_scale)
                                   .expand([1]).to_event(1))

    def forward(self, x, y=None):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        mean = self.out(x).squeeze(-1)
        sigma = 0.1  # fixed likelihood std
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean