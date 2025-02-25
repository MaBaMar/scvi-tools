"""streamlined DIVA implementation for RQM wrapping"""
from __future__ import annotations

import logging
from typing import Literal, Callable

import numpy as np
import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial, Poisson
from scvi.module._diva import DIVA
from scvi.module.base import LossOutput, auto_move_data
from scvi.nn import one_hot
from torch.distributions import kl_divergence as kl, Normal

logger = logging.getLogger(__name__)


class RQMDiva(DIVA):
    _pred_func: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, *args,
                 label_generator: Literal['prior_based', 'internal_classifier'],
                 conservativeness: float,
                 **kwargs):
        """

        Parameters
        ----------
        args
        label_generator
        conservativeness This value defines how heavily unlabeled samples are considered. The supervised part of the loss is weighted by
                         `conservativeness` while the unsupervised part of the loss is weighted by (1-`conservativeness`).
        kwargs
        """
        super().__init__(*args, **kwargs)
        if conservativeness > 1 or conservativeness < 0:
            raise ValueError("conservativeness must be a value between 0 and 1")
        if conservativeness in [0, 1]:
            logger.warn(f"conservativeness is: {conservativeness}. Hence, not all training samples might be considered, potentially "
                        f"affecting the batch size. It is better to remove unwanted training samples from training data beforehand.")

        self.conservativeness = conservativeness
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
        y: torch.Tensor,
        library: torch.Tensor,
        size_factor=None,
    ) -> dict[str, torch.Tensor | torch.distributions.Distribution]:
        has_no_label = (y == (self.n_labels - 1)).flatten()
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

        marginal_pseudo_weight = torch.ones((zy_x.shape[0],), device=self.device, dtype=torch.float)

        # generate pseudo labels for samples without labels
        zy_x_unlabeled = zy_x[has_no_label]
        with torch.no_grad():
            self.eval()  # TODO: analyze impact
            probs = self._pred_func(zy_x_unlabeled)
            self.train()  # TODO: analyze impact
        p_y_zy = torch.distributions.Categorical(probs)
        y_pseudo = probs.argmax(dim=-1).view(-1, 1)
        marginal_pseudo_weight[has_no_label] = torch.exp(p_y_zy.log_prob(y_pseudo.flatten()).view(-1, ))

        y[has_no_label] = y_pseudo

        p_zy_y = None
        if self.use_learnable_priors:
            p_zd_d, _ = self.prior_zd_d_encoder(one_hot(d, self.n_batch))
            if not self._unsupervised:
                p_zy_y, _ = self.prior_zy_y_encoder(one_hot(y, self.n_labels - 1))
        else:
            p_zd_d = Normal(torch.zeros_like(zd_x), torch.ones_like(zd_x))
            if not self._unsupervised:
                p_zy_y = Normal(torch.zeros_like(zy_x), torch.ones_like(zy_x))

        return {'px_recon': px_recon, 'p_zd_d': p_zd_d, 'p_zy_y': p_zy_y, 'marginal_pseudo_weight': marginal_pseudo_weight,
                'has_no_label': has_no_label}

    def _weight_conservativeness(self, tensor: torch.Tensor, has_no_label: torch.Tensor) -> torch.Tensor:
        mult_vec = torch.ones_like(tensor) * self.conservativeness
        mult_vec[has_no_label] = (1 - self.conservativeness)
        return tensor * mult_vec

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,  # epsilon
        **kwargs
    ) -> LossOutput:

        x = tensors[REGISTRY_KEYS.X_KEY]
        has_no_label = generative_outputs['has_no_label']

        # avoid repeating the indexing process
        neg_reconstruction_loss = -generative_outputs["px_recon"].log_prob(x).sum(-1)

        # KL-loss
        kl_zd = kl(
            inference_outputs['q_zd_x'],
            generative_outputs['p_zd_d'],
        ).sum(dim=-1)
        # kl_zd = self._weight_conservativeness(kl_zd, has_no_label)

        if not self.use_observed_lib_size:
            kl_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
            # kl_l = self._weight_conservativeness(kl_l, has_no_label)
        else:
            kl_l = torch.tensor(0.0, device=x.device)

        marginal_pseudo_weight = generative_outputs['marginal_pseudo_weight']

        kl_zy = marginal_pseudo_weight * (
                inference_outputs['q_zy_x'].log_prob(inference_outputs['zy_x']).sum(-1) - generative_outputs['p_zy_y'].log_prob(
                inference_outputs['zy_x']))
        # kl_zy = self._weight_conservativeness(kl_zy, has_no_label)

        # KL loss
        kl_local_for_warmup = self.beta_d * kl_zd + self.beta_y * kl_zy

        # auxiliary_loss = self.alpha_y * F.cross_entropy(prob_all[~has_no_label], y[~has_no_label].flatten()) * self.conservativeness

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_l

        loss_sum = neg_reconstruction_loss + weighted_kl_local

        no_label_percentage = has_no_label.mean(dtype=float)
        if no_label_percentage < 1:
            loss_supervised = loss_sum[~has_no_label].mean()
        else:
            loss_supervised = 0
        if no_label_percentage > 0:
            loss_unsupervised = loss_sum[has_no_label].mean()
        else:
            loss_unsupervised = 0

        loss = loss_supervised * self.conservativeness + loss_unsupervised * (1 - self.conservativeness)
        kl_local = {
            'kl_divergence_l': kl_l,
            'kl_divergence_zd': kl_zd,
            'kl_divergence_zy': kl_zy,
        }

        return LossOutput(loss, neg_reconstruction_loss, kl_local, extra_metrics={'kl_l': kl_l.mean(), 'kl_zd': kl_zd.mean(),
                                                                                  'kl_zy': kl_zy.mean(),
                                                                                  'loss_unsupervised': loss_unsupervised,
                                                                                  'loss_supervised': loss_supervised,
                                                                                  'no_label_percentage': no_label_percentage,
                                                                                  'loss_sum_mean': loss_sum.mean()})

    @torch.inference_mode()
    def sample_reconstruction_ll(
        self,
        tensors,
        n_mc_samples,
        n_mc_samples_per_pass=1,
    ):
        """Computes the marginal log likelihood of the model.

        Parameters
        ----------
        tensors
            Dict of input tensors, typically corresponding to the items of the data loader.
        n_mc_samples
            Number of Monte Carlo samples to use for the estimation of the marginal log likelihood.
        n_mc_samples_per_pass
            Number of Monte Carlo samples to use per pass. This is useful to avoid memory issues.
        """
        # auto move did not work for directory
        for key in tensors.keys():
            tensors[key] = tensors[key].to(self.device)

        p_x_zl_sum = []
        if n_mc_samples_per_pass > n_mc_samples:
            logger.warn(
                "Number of chunks is larger than the total number of samples, setting it to the number of samples"
            )
            n_mc_samples_per_pass = n_mc_samples
        n_passes = int(np.ceil(n_mc_samples / n_mc_samples_per_pass))
        for _ in range(n_passes):
            # Distribution parameters and sampled variables
            _, _, losses = self.forward(
                tensors, inference_kwargs={"n_samples": n_mc_samples_per_pass}
            )

            # Reconstruction Loss
            p_x_zl = losses.dict_sum(losses.reconstruction_loss)
            p_x_zl_sum.append(p_x_zl.cpu())
        p_x_zl_sum = torch.logsumexp(torch.stack(p_x_zl_sum), dim=0)

        return p_x_zl_sum
