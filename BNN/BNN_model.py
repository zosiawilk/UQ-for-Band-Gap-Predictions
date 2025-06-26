import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch
import torch.nn as nn
import torch.nn.functional as F

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
