import torch
from torch.distributions import Normal
import torch.distributions as dist
from scvi.priors.base_prior import BasePrior
import normflows as nf
from scvi.autotune._types import Tunable

class NormalFlow(BasePrior):
    def __init__(self, n_latent: int, num_layers: Tunable[int] = 8, flow = None):
        super(NormalFlow, self).__init__()
        self.base = nf.distributions.base.DiagGaussian(n_latent)
        self.flows = []
        if flow is None:
            param_map = nf.nets.MLP([int(n_latent/2),64,64,n_latent], init_zeros=True)
            flow = [nf.flows.AffineCouplingBlock(param_map),nf.flows.Permute(n_latent,mode='swap')]

        for i in range(num_layers):
            self.flows.extend(flow)   
        self.dist = nf.NormalizingFlow(self.base,self.flows)

    @property
    def distribution(self):
        return self.dist

    def sample(self, n_samples: int):
        return self.distribution.sample(n_samples)

    def log_prob(self, z):
        return self.distribution.log_prob(z)