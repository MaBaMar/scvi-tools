"""streamlined DIVA implementation for RQM wrapping"""
from __future__ import annotations

from typing import Literal

import torch
from torch.distributions import kl_divergence as kl
from torch.nn import functional as F

from scvi import REGISTRY_KEYS
from scvi._types import Tunable
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial, Poisson
from scvi.module._diva_unsup import TunedDIVA
from scvi.module.base import LossOutput, auto_move_data
from scvi.nn import one_hot
from scvi.nn._base_components import DecoderRQM


class RQMDiva(TunedDIVA):

    def __init__(self,
                 n_input: int,
                 n_batch: int,
                 n_labels: int,
                 n_latent_d: int = 4,
                 n_latent_y: int = 10,
                 beta_d: float = 10,
                 beta_y: float = 10,
                 alpha_d: float = 100,
                 alpha_y: float = 100,
                 priors_n_hidden: int = 32,
                 priors_n_layers: int = 1,
                 posterior_n_hidden: int = 128,
                 posterior_n_layers: int = 1,
                 decoder_n_hidden: int = 128,
                 decoder_n_layers: int = 1,
                 dropout_rate: float = 0.1,
                 lib_encoder_n_hidden: int = 128,
                 lib_encoder_n_layers: int = 1,
                 dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
                 log_variational: Tunable[bool] = True,
                 gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
                 latent_distribution: Literal["normal", "ln"] = "normal",
                 use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
                 use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
                 ):

        super().__init__(*locals())

        """reconstruction term p_theta(x|zd, zy) -> overwrite it with RQM compatible decoder"""
        self.reconst_dxy_decoder = DecoderRQM(
            n_input_y=n_latent_y,
            n_input_d=n_latent_d,
            n_output=n_input,  # output dim of decoder = original input dim
            n_layers=decoder_n_layers,
            n_hidden=decoder_n_hidden,
            use_batch_norm=use_batch_norm == "decoder" or use_batch_norm == "both",
            use_layer_norm=use_layer_norm == "decoder" or use_layer_norm == "both",
        )

    def _get_inference_input(self, tensors: dict[str, torch.Tensor], **kwargs):
        return {'x': tensors[REGISTRY_KEYS.X_KEY], 'd': tensors[REGISTRY_KEYS.BATCH_KEY],
                'y': tensors[REGISTRY_KEYS.LABELS_KEY]}

    def _get_generative_input(self, tensors: dict[str, torch.Tensor], inference_outputs: dict[str, torch.Tensor],
                              **kwargs):
        return {'d': tensors[REGISTRY_KEYS.BATCH_KEY], **inference_outputs}

    @auto_move_data
    def inference(
        self,
        x,
        d,
        y,
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

        px_scale, px_r, px_rate, px_dropout = self.reconst_dxy_decoder.forward(
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

        p_zd_d, _ = self.prior_zd_d_encoder(one_hot(d, self.n_batch))
        p_zy_y, _ = self.prior_zy_y_encoder(one_hot(y, self.n_labels))

        return {'px_recon': px_recon, 'd_hat': d_hat, 'y_hat': y_hat, 'p_zd_d': p_zd_d, 'p_zd_y': p_zy_y}

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
        kl_zy = kl(
            inference_outputs['q_zy_x'],
            generative_outputs['p_zy_y'],
        ).sum(dim=1)

        # KL loss
        kl_local_for_warmup = (self.beta_d * kl_zd * torch.tensor(self._kl_weights_d, device=self.device)[d]
                               + self.beta_y * kl_zy * torch.tensor(self._kl_weights_y, device=self.device)[y])

        aux_loss = (
            self.alpha_d * (aux_d := F.cross_entropy(generative_outputs["d_hat"], d.view(-1, )))
            + self.alpha_y * (
                aux_y := F.cross_entropy(generative_outputs["y_hat"], y.view(-1, ), weight=self._ce_weights_y.to(self.device)))
        )
        extra_metrics = {
            'aux_d': aux_d,
            'aux_y': aux_y,
        }

        weighted_kl_local = kl_weight * kl_local_for_warmup

        loss = torch.mean(neg_reconstruction_loss + weighted_kl_local) + ce_weight * aux_loss

        kl_local = {
            'kl_divergence_zd': kl_zd,
            'kl_divergence_zy': kl_zy,
        }

        return LossOutput(loss, neg_reconstruction_loss, kl_local, extra_metrics=extra_metrics)
