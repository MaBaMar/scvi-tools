"""streamlined DIVA implementation for RQM wrapping"""
from __future__ import annotations

import torch
from scvi import REGISTRY_KEYS
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial, Poisson
from scvi.module._diva import DIVA
from scvi.module.base import LossOutput, auto_move_data
from scvi.nn import one_hot
from torch.distributions import kl_divergence as kl
from torch.nn import functional as F


class RQMDiva(DIVA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_inference_input(self, tensors: dict[str, torch.Tensor], **kwargs):
        return {'x': tensors[REGISTRY_KEYS.X_KEY], 'd': tensors[REGISTRY_KEYS.BATCH_KEY]}

    def _get_generative_input(self, tensors: dict[str, torch.Tensor], inference_outputs: dict[str, torch.Tensor],
                              **kwargs):
        return {'d': tensors[REGISTRY_KEYS.BATCH_KEY], **inference_outputs}

    @auto_move_data
    def inference(
        self,
        x,
        d,
        n_samples: int = 1,
        **kwargs
    ) -> dict[str, torch.Tensor | torch.distributions.Distribution]:

        if self.log_variational:
            x = torch.log(1 + x)

        q_zd_x, zd_x = self.posterior_zd_x_encoder(x)
        q_zy_x, zy_x = self.posterior_zy_x_encoder(x)

        library = torch.log(x.sum(1)).unsqueeze(1)

        if n_samples > 1:
            utran_zd = q_zd_x.rsample((n_samples,))
            utran_zy = q_zy_x.rsample((n_samples,))
            zd_x = self.posterior_zd_x_encoder.z_transformation(utran_zd)
            zy_x = self.posterior_zy_x_encoder.z_transformation(utran_zy)
            library = library.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )

        outputs = {'zd_x': zd_x, 'zy_x': zy_x, 'q_zd_x': q_zd_x, 'q_zy_x': q_zy_x, 'library': library}
        return outputs

    @auto_move_data
    def generative(
        self,
        d: torch.Tensor,
        zd_x: torch.Tensor, zy_x: torch.Tensor,
        library: torch.Tensor,
        **kwargs
    ) -> dict[str, torch.Tensor | torch.distributions.Distribution]:

        px_scale, px_r, px_rate, px_dropout = self.reconstruction_dxy_decoder.forward(
            self.dispersion,
            zy_x,
            zd_x,
            library
        )

        if self.dispersion == "gene":
            px_r = self.px_r
        else:
            raise NotImplementedError("Currently only supporting `gene` dispersion")

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

        # priors
        d_hat = self.aux_d_zd_enc(zd_x)
        y_hat = self.aux_y_zy_enc(zy_x)

        y = torch.argmax(y_hat.clone(), dim=1).view(-1, 1)

        p_zd_d, _ = self.prior_zd_d_encoder(one_hot(d, self.n_batch))
        p_zy_y, _ = self.prior_zy_y_encoder(one_hot(y, self.n_labels))

        return {'px_recon': px_recon, 'd_hat': d_hat, 'y_hat': y_hat, 'p_zd_d': p_zd_d, 'p_zy_y': p_zy_y}

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,  # epsilon
        ce_weight: float = 1.0,
        **kwargs
    ) -> LossOutput:

        x = tensors[REGISTRY_KEYS.X_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]
        d = tensors[REGISTRY_KEYS.BATCH_KEY]

        # avoid repeating the indexing process
        neg_reconstruction_loss = -generative_outputs["px_recon"].log_prob(x).sum(-1)

        # KL-loss
        kl_zd = kl(
            inference_outputs['q_zd_x'],
            generative_outputs['p_zd_d'],
        ).sum(dim=1)
        kl_zy = (inference_outputs['q_zy_x'].log_prob(inference_outputs['zy_x']) - generative_outputs['p_zy_y'].log_prob(
            inference_outputs['zy_x'])).sum(-1).view(-1, 1)

        # KL loss
        kl_local_for_warmup = (self.beta_d * kl_zd * self._kl_weights_d.to(self.device)[d]
                               + self.beta_y * kl_zy * self._kl_weights_y.to(self.device)[y])

        weighted_kl_local = kl_weight * kl_local_for_warmup

        loss = torch.mean(neg_reconstruction_loss + weighted_kl_local)

        kl_local = {
            'kl_divergence_zd': kl_zd,
            'kl_divergence_zy': kl_zy,
        }

        return LossOutput(loss, neg_reconstruction_loss, kl_local)
