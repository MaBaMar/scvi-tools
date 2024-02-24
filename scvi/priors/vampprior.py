import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from scvi.nn import Encoder
from scvi.priors.base_prior import BasePrior
from scvi.autotune._types import Tunable

class VampPrior(BasePrior):
    def __init__(self, n_latent: int, n_input: int, encoder: Encoder,
                 n_components: Tunable[int] = 50, n_hidden: Tunable[int] = 256):
        super(VampPrior, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.encoder = encoder
        self.register_buffer("pseudo_inputs", torch.eye(n_components))
        self.pseudo_transfromer = nn.Sequential(*[
            nn.Linear(n_components, 256),
            nn.ReLU(),
            nn.Linear(256, n_input),
            nn.ReLU()
        ])

        self.w = nn.Parameter(torch.zeros(self.pseudo_inputs.shape[0], ))

    @property
    def distribution(self):
        inputs = self.pseudo_transfromer(self.pseudo_inputs)
        comp, _ = self.encoder(inputs)
        comp = dist.Independent(comp, 1)
        mix = dist.Categorical(F.softmax(self.w, dim=0))
        return dist.MixtureSameFamily(mix, comp)

    def sample(self, n_samples: int):
        return self.distribution.sample((n_samples,))

    def log_prob(self, z):
        return self.distribution.log_prob(z)
    
    def description(self):
        return "Vamp Prior with pseudo inputs: " + str(self.pseudo_inputs) + " and mixing parameter : " + str(self.w)

    def get_pseudo_inputs(self):
        psi = self.pseudo_transfromer(self.pseudo_inputs)
        return psi, self.encoder(psi)