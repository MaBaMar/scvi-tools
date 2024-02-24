import torch
from torch.distributions import Normal
import torch.distributions as dist
from scvi.priors.base_prior import BasePrior
import torch.nn.functional as F
from scvi.autotune._types import Tunable


class MixOfGausPrior(BasePrior):
    def __init__(self, n_latent: int, k : Tunable[int] = 50):
        super(MixOfGausPrior, self).__init__()
        self.k = k
        self.w = torch.nn.Parameter(torch.zeros(k,))
        self.mean = torch.nn.Parameter(torch.randn((k,n_latent)))
        self.logvar = torch.nn.Parameter(torch.randn((k,n_latent)))
    @property
    def distribution(self):
        comp = Normal(self.mean,torch.exp(0.5 * self.logvar))
        comp = dist.Independent(comp,1)
        mix = dist.Categorical(F.softmax(self.w, dim=0))
        return dist.MixtureSameFamily(mix, comp)

    def sample(self, n_samples: int):
        return self.distribution.sample((n_samples,))

    def log_prob(self, z):
        return self.distribution.log_prob(z)
    
    def description(self):
        return "Mixture of Gaussians with w: " +str(self.w)+ "Prior with means: " + str(self.mean) + " and log variance: " + str(self.logvar)
