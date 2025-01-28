"""streamlined DIVA implementation for RQM wrapping"""
from __future__ import annotations

import logging
from typing import Literal, Callable

import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial, Poisson
from scvi.module._diva import DIVA
from scvi.module.base import LossOutput, auto_move_data
from scvi.nn import one_hot
from torch.distributions import kl_divergence as kl, OneHotCategorical, Independent, Normal

logger = logging.getLogger(__name__)

class RQMDiva(DIVA):

    _pred_func: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, *args, label_generator: Literal['prior_based', 'internal_classifier'], **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"[RQMDiva] using pred_type: {label_generator}")
        match label_generator:
            case 'prior_based':
                self._pred_func = self._pred_prior
            case 'internal_classifier':
                self._pred_func = self._pred_cls
            case _:
                raise ValueError(f'Invalid pred_type {label_generator}')

    # None of the inference methods are changed, as inference is identical when training on reference and query data

    @auto_move_data
    def generative(
        self,
        zd_x: torch.Tensor,
        zy_x: torch.Tensor,
        d: torch.Tensor,
        library: torch.Tensor,
        size_factor=None,
        **kwargs
    ) -> dict[str, torch.Tensor | torch.distributions.Distribution]:

        if not self.use_size_factor_key:
            size_factor = library

        px_scale, px_r, px_rate, px_dropout = self.reconstruction_dxy_decoder(
            self.dispersion,
            zy_x,
            zd_x,
            size_factor
        )

        if self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(d, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        else:
            raise NotImplementedError("Currently only supporting `gene` or `gene-batch` dispersion")

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px_recon = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px_recon = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px_recon = Poisson(px_rate, scale=px_scale)
        else:
            raise ValueError(
                f"gene_likelihood must be one of ['zinb', 'nb','poisson'], but input was {self.gene_likelihood}"
            )

        self.eval() # TODO: analyze impact
        probs = self._pred_func(zy_x)
        self.train() # TODO: analyze impact
        p_y_zy = torch.distributions.Categorical(probs)
        y = p_y_zy.sample().view(-1, 1)

        # priors
        p_zy_y = None
        if self.use_learnable_priors:
            p_zd_d, _ = self.prior_zd_d_encoder(one_hot(d, self.n_batch))
            if not self._unsupervised:
                p_zy_y, _ = self.prior_zy_y_encoder(one_hot(y, self.n_labels))
        else:
            p_zd_d = Normal(torch.zeros_like(zd_x), torch.ones_like(zd_x))
            if not self._unsupervised:
                p_zy_y = Normal(torch.zeros_like(zy_x), torch.ones_like(zy_x))

        return {'px_recon': px_recon, 'p_zd_d': p_zd_d, 'p_zy_y': p_zy_y, 'p_y_zy': p_y_zy, 'y_pred': y}

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,  # epsilon
        **kwargs
    ) -> LossOutput:

        x = tensors[REGISTRY_KEYS.X_KEY]

        # avoid repeating the indexing process
        neg_reconstruction_loss = -generative_outputs["px_recon"].log_prob(x).sum(-1)

        # KL-loss
        kl_zd = kl(
            inference_outputs['q_zd_x'],
            generative_outputs['p_zd_d'],
        ).sum(dim=1)

        if not self.use_observed_lib_size:
            kl_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_l = torch.tensor(0.0, device=x.device)

        p_y_zy = generative_outputs['p_y_zy']
        y_pred = generative_outputs['y_pred']

        marginal_weight = torch.exp(p_y_zy.log_prob(y_pred.reshape(y_pred.shape[0],)).view(-1,))

        kl_zy = marginal_weight * (inference_outputs['q_zy_x'].log_prob(inference_outputs['zy_x']) - generative_outputs['p_zy_y'].log_prob(
            inference_outputs['zy_x'])).sum(-1).view(-1,)

        # KL loss
        kl_local_for_warmup = self.beta_d * kl_zd + self.beta_y * kl_zy

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_l

        loss = torch.mean(neg_reconstruction_loss + weighted_kl_local)

        kl_local = {
            'kl_divergence_l': kl_l,
            'kl_divergence_zd': kl_zd,
            'kl_divergence_zy': kl_zy,
        }

        return LossOutput(loss, neg_reconstruction_loss, kl_local, extra_metrics={'kl_l': kl_l.mean(), 'kl_zd': kl_zd.mean(), 'kl_zy': kl_zy.mean()})
