"""Semi-Supervised scDIVA implementation"""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi._types import Tunable
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial, Poisson
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot
from sklearn.utils import compute_class_weight
from torch import nn
from torch.distributions import kl_divergence as kl
from torch.nn import functional as F


class TunedDIVA(BaseModuleClass):

    def __init__(self,
                 n_input: int,
                 n_batch: int,
                 n_labels: int,
                 n_latent_d: int = 4,  # 2, 5, 10
                 n_latent_y: int = 10,
                 beta_d: float = 1,  # 1, 10, 50
                 beta_y: float = 1,
                 alpha_d: float = 1000,
                 alpha_y: float = 1500,
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

        super().__init__()

        if n_latent_y <= 0:
            raise ValueError("'n_latent_y' must be positive'")

        self.n_input = n_input
        self.n_batch = n_batch
        self.n_labels = n_labels

        self.n_latent_d = n_latent_d
        self.n_latent_y = n_latent_y

        self.dropout_rate = dropout_rate
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution

        # library stuff (want to stick close to SCVI in principle)
        self.use_observed_lib_size = True

        self.beta_d = beta_d
        self.alpha_d = alpha_d

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        """reconstruction term p_theta(x|zd, zy)"""
        self.reconst_dxy_decoder = DecoderSCVI(
            n_input=n_latent_d + n_latent_y,
            n_output=n_input,  # output dim of decoder = original input dim
            n_layers=decoder_n_layers,
            n_hidden=decoder_n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

        """model priors p_theta(zd|d), p_theta(zy|y)"""
        self.prior_zd_d_encoder = Encoder(
            n_input=n_batch,
            n_output=n_latent_d,
            n_layers=priors_n_layers,
            n_hidden=priors_n_hidden,
            dropout_rate=self.dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True
        )

        self.prior_zy_y_encoder = Encoder(
            n_input=n_labels,
            n_output=n_latent_y,
            n_layers=priors_n_layers,
            n_hidden=priors_n_hidden,
            dropout_rate=self.dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
        )

        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=lib_encoder_n_layers,
            n_hidden=lib_encoder_n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
        )

        """variational posteriors q_phi(zd|x), q_phi(zy|x)"""
        self.posterior_zd_x_encoder = Encoder(
            n_input=n_input,
            n_output=n_latent_d,
            n_layers=posterior_n_layers,
            n_hidden=posterior_n_hidden,
            distribution=self.latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
        )

        self.beta_y = beta_y
        self.alpha_y = alpha_y
        self.posterior_zy_x_encoder = Encoder(
            n_input=n_input,
            n_output=n_latent_y,
            n_layers=posterior_n_layers,
            n_hidden=posterior_n_hidden,
            distribution=self.latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
        )

        """auxiliary tasks q_w(y|zy) and q_w(d|zd)"""
        self.aux_y_zy_enc = torch.nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_latent_y, n_labels)
        )

        self.aux_d_zd_enc = torch.nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_latent_d, n_batch)
        )

    def _get_inference_input(self, tensors: dict[str, torch.Tensor], **kwargs):
        return {'x': tensors[REGISTRY_KEYS.X_KEY], 'd': tensors[REGISTRY_KEYS.BATCH_KEY],
                'y': tensors[REGISTRY_KEYS.LABELS_KEY]}

    def _get_generative_input(self, tensors: dict[str, torch.Tensor], inference_outputs: dict[str, torch.Tensor],
                              **kwargs):
        return {'d': tensors[REGISTRY_KEYS.BATCH_KEY], 'y': tensors[REGISTRY_KEYS.LABELS_KEY], **inference_outputs}

    @auto_move_data
    def inference(self,
                  x,
                  d,
                  y,
                  n_samples: int = 1,
                  **kwargs) -> dict[str, torch.Tensor | torch.distributions.Distribution]:
        """normal forward pass -> works the same for both supervised AND unsupervised samples
        zd_x, zx_x, zy_x, q_zd_x, q_zx_x, q_zy_x, library"""

        has_no_label, has_label = self._get_label_masks(y)

        library = torch.empty((x.shape[0], 1), device=self.device, dtype=torch.float)
        zd_x = torch.empty((x.shape[0], self.n_latent_d), device=self.device, dtype=torch.float)
        zy_x = torch.empty((x.shape[0], self.n_latent_y), device=self.device, dtype=torch.float)

        library[has_label] = torch.log(x[has_label, :].sum(1)).unsqueeze(1)
        library[has_no_label] = torch.log(x[has_no_label, :].sum(1)).unsqueeze(1)
        if self.log_variational:
            x = torch.log(1 + x)

        x_sup = x[has_label, :]
        x_unsup = x[has_no_label, :]

        # tuples of posterior distribution and drawn latent variable
        q_zd_x, zd_x = self.posterior_zd_x_encoder(x)  # .copy() ?!?
        q_zy_x_sup, zy_x[has_label] = self.posterior_zy_x_encoder(x_sup)

        # inference unsup
        q_zy_x_unsup, zy_x[has_no_label] = self.posterior_zy_x_encoder(x_unsup)

        if n_samples > 1:
            utran_zd = q_zd_x.rsample((n_samples,))
            zd_x = self.posterior_zd_x_encoder.z_transformation(utran_zd)
            utran_zy = torch.empty((n_samples, *zy_x.shape), device=self.device, dtype=torch.float)
            utran_zy[:, has_label, :] = q_zy_x_sup.rsample((n_samples,))
            utran_zy[:, has_no_label, :] = q_zy_x_unsup.rsample((n_samples,))
            zy_x = self.posterior_zy_x_encoder.z_transformation(utran_zy)
        outputs = {
            'zd_x': zd_x, 'zy_x': zy_x, 'library': library, 'q_zd_x': q_zd_x, 'q_zy_x_sup': q_zy_x_sup,
            'q_zy_x_unsup': q_zy_x_unsup, 'has_label': has_label, 'has_no_label': has_no_label
        }
        return outputs

    @auto_move_data
    def _get_label_masks(self, y) -> (torch.Tensor, torch.Tensor):
        """correct use: has_no_label, has_label = self._get_label_masks(y)"""
        return (t := (y == self.n_labels).reshape(-1, )), ~t

    @auto_move_data
    def generative(
        self,
        d: torch.Tensor, y: torch.Tensor,
        zd_x: torch.Tensor, zy_x: torch.Tensor,
        library: torch.Tensor,
        has_label: torch.Tensor,
        has_no_label: torch.Tensor,
        **kwargs
    ) -> dict[str, torch.Tensor | torch.distributions.Distribution]:

        px_scale, px_r, px_rate, px_dropout = self.reconst_dxy_decoder(
            self.dispersion,
            torch.cat([zd_x, zy_x], dim=-1),
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

        y[has_no_label] = torch.argmax(y_hat[has_no_label], dim=1).view(-1, 1)

        p_zd_d, _ = self.prior_zd_d_encoder(one_hot(d, self.n_batch))
        p_zy_y_unsup, _ = self.prior_zy_y_encoder(one_hot(y[has_no_label], self.n_labels))
        p_zy_y_sup, _ = self.prior_zy_y_encoder(one_hot(y[has_label], self.n_labels))

        return {'px_recon': px_recon, 'd_hat': d_hat, 'y_hat': y_hat, 'p_zd_d': p_zd_d, 'p_zy_y_sup': p_zy_y_sup,
                'p_zy_y_unsup': p_zy_y_unsup, 'y': y}

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
        y = generative_outputs['y']
        d = tensors[REGISTRY_KEYS.BATCH_KEY]

        has_no_label, has_label = inference_outputs['has_no_label'], inference_outputs['has_label']

        # avoid repeating the indexing process
        d_sup = d[has_label, :]
        d_unsup = d[has_no_label, :]
        y_sup = y[has_label, :]

        # reconstruction_loss
        neg_reconstruction_loss = -generative_outputs["px_recon"].log_prob(x).sum(-1)
        kl_zy = torch.empty_like(y, dtype=torch.float, device=y.device)

        # KL-loss
        kl_zd = kl(
            inference_outputs['q_zd_x'],
            generative_outputs['p_zd_d'],
        ).sum(dim=1)
        kl_zy[has_label] = kl(
            inference_outputs['q_zy_x_sup'],
            generative_outputs['p_zy_y_sup'],
        ).sum(dim=1).view(-1, 1)

        kl_zy[has_no_label] = - (
            generative_outputs['p_zy_y_unsup'].log_prob(inference_outputs['zy_x'][has_no_label]).sum(-1)
            - inference_outputs['q_zy_x_unsup'].log_prob(inference_outputs['zy_x'][has_no_label]).sum(-1)).view(-1,
                                                                                                                1)
        # TODO: add weighting here!
        weighted_kl_local = kl_weight * (
            self.beta_d * kl_zd * self._kl_weights_d.to(self.device)[d] + self.beta_y * kl_zy *
            self._kl_weights_y.to(self.device)[y])

        # auxiliary regularizer:
        # TODO: include?!?

        extra_metrics = {}

        # auxiliary losses from the classifiers
        # 1. bach loss
        aux_d = F.cross_entropy(generative_outputs['d_hat'], d.view(-1, ))
        extra_metrics["aux_d"] = aux_d

        # 2. cell-type loss
        aux_y_sup = F.cross_entropy(generative_outputs['y_hat'], y.view(-1, ),
                                    weight=self._ce_weights_y.to(self.device))
        extra_metrics["aux_y"] = aux_y_sup

        aux_loss = self.alpha_d * aux_d + self.alpha_y * aux_y_sup

        final_loss = torch.mean(neg_reconstruction_loss + weighted_kl_local) + ce_weight * aux_loss

        kl_local = {
            "kl_divergence_zd": kl_zd,
            "kl_divergence_zy": kl_zy
        }

        return LossOutput(final_loss, neg_reconstruction_loss, kl_local, extra_metrics=extra_metrics)

    def sample(self, *args, **kwargs):
        pass

    def init_ce_weight_y(self, adata: AnnData, indices, label_key: str):
        # BEWARE!!! Validation loss will use training weighting for CE!
        """Does not work if train split does not contain all celltypes"""
        train_adata_labels = adata[indices].obs[label_key]
        cat_y = train_adata_labels[train_adata_labels != "unknown"].cat.codes
        self._ce_weights_y = torch.tensor(
            compute_class_weight('balanced', classes=np.array(range(self.n_labels)), y=cat_y), dtype=torch.float)

    def init_kl_weights(self, adata: AnnData, indices, label_key: str, batch_key: str):
        train_adata_y_labels = adata[indices].obs[label_key]
        train_adata_d_labels = adata[indices].obs[batch_key]

        cat_y = train_adata_y_labels[train_adata_y_labels != "unknown"].cat.codes
        cat_d = train_adata_d_labels.cat.codes

        cat_d_holdout = np.setdiff1d(np.unique(adata.obs[batch_key].cat.codes), np.unique(cat_d))
        cat_y_holdout = np.setdiff1d(np.unique((t := adata.obs[label_key])[t != "unknown"].cat.codes), cat_y)

        """Note: We might have batches only present in validation set. This needs to be taken into consideration here.
        They'll get very high training weights, but since we do not use those batches during training, that does
        not matter. However, the high values might corrupt validation loss. Therefore, we set such batches to 1"""

        """use # samples / (1 + # classes * # bincount) to allow calculation for empty batches (here 1+ in divisor)"""
        cat_d_corrected = np.append(cat_d, range(self.n_batch))
        cat_y_corrected = np.append(cat_y, range(self.n_labels))

        self._kl_weights_y = torch.from_numpy(
            compute_class_weight('balanced', classes=np.array(range(self.n_labels)), y=cat_y_corrected))

        self._kl_weights_d = torch.from_numpy(
            compute_class_weight('balanced', classes=np.array(range(self.n_batch)), y=cat_d_corrected))

        self._kl_weights_y[cat_y_holdout] = 1
        self._kl_weights_d[cat_d_holdout] = 1
