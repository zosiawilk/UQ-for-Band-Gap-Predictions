import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class BayesianGCN(PyroModule):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # Bayesian GCN layer 1
        self.conv1 = PyroModule[GCNConv](in_channels, hidden_channels)
        self.conv1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_channels, in_channels]).to_event(2))
        self.conv1.bias   = PyroSample(dist.Normal(0., 1.).expand([hidden_channels]).to_event(1))

        # Bayesian GCN layer 2
        self.conv2 = PyroModule[GCNConv](hidden_channels, hidden_channels)
        self.conv2.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_channels, hidden_channels]).to_event(2))
        self.conv2.bias   = PyroSample(dist.Normal(0., 1.).expand([hidden_channels]).to_event(1))

        # Final linear Bayesian layer
        self.fc_out = PyroModule[torch.nn.Linear](hidden_channels, out_channels)
        self.fc_out.weight = PyroSample(dist.Normal(0., 1.).expand([out_channels, hidden_channels]).to_event(2))
        self.fc_out.bias   = PyroSample(dist.Normal(0., 1.).expand([out_channels]).to_event(1))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)

        mu = self.fc_out(x)  # shape: [num_graphs, 1]
        mu = mu.view(-1)     # final shape: [num_graphs]
        mu = self.fc_out(x).view(-1)      # [num_graphs]
        pyro.deterministic("mu_det", mu)  # <── expose raw mean

        num_graphs = mu.size(0)

        # Sample or fix the uncertainty (sigma)
        sigma = pyro.sample("sigma_graph", dist.Uniform(0.05, 0.3))
        sigma = sigma.expand(num_graphs)

        target = None
        if hasattr(data, "y") and data.y is not None:
            target = data.y.view(-1)  # shape must match mu

        with pyro.plate("data", size=num_graphs):
            pyro.sample("mu_graph", dist.Normal(mu, sigma), obs=target)

        return mu
