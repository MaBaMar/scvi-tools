"""DIVA implementation supervised"""
from __future__ import annotations

import warnings
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi._types import Tunable
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial, Poisson
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl


# TODO: support minification
class DIVA(BaseModuleClass):
    """Slight modification of DIVA model to SCVI setting. Additionally, uses library size just as in SCVI,
    i.e. generative is p(x|zd,zy,zx,l) where zd is batch-latent representation, zy is label-latent representation,
    zx is 'other effects' latent-space and l models the library size (in our settings, we use observed l, not l
    as probability distribution) """

    _ce_weights_y = None
    _ce_weights_d = None

    def __init__(
        self,
        n_input: int,
        n_batch: int,
        n_labels: int,
        n_latent_d: int = 4,
        n_latent_x: int = 4,
        n_latent_y: int = 10,
        beta_d: float = 1,
        beta_x: float = 13,
        beta_y: float = 1,
        alpha_d: float = 1000,
        alpha_y: float = 1500,
        priors_n_hidden: int = 32,
        priors_n_layers: int = 1,
        posterior_n_hidden: int = 128,
        posterior_n_layers: int = 1,
        encoder_zd_x_layers: int = 1,
        encoder_zd_x_hidden: int = 128,
        decoder_n_hidden: int = 128,
        decoder_n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: Tunable[bool] = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_linear_decoder: bool = False,
        use_size_factor_key: bool = False,
        use_observed_lib_size: Tunable[bool] = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        extra_encoder_kwargs: Optional[dict] = None,
        extra_decoder_kwargs: Optional[dict] = None
    ):
        super().__init__()

        self.n_input = n_input
        self.n_batch = n_batch
        self.n_labels = n_labels

        self.dropout_rate = dropout_rate
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution

        # library stuff (want to stick close to SCVI in principle)
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        self.beta_d = beta_d
        self.beta_x = beta_x
        self.beta_y = beta_y

        self.alpha_d = alpha_d
        self.alpha_y = alpha_y

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

        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        _extra_encoder_kwargs = extra_encoder_kwargs or {}

        """reconstruction term p_theta(x|zd, zx, zy)"""

        # We feed the parameters to the decoder as [d, x, y] TODO: use in forward
        self.reconst_dxy_decoder = DecoderSCVI(
            n_input=n_latent_d + n_latent_x + n_latent_y,
            n_output=n_input,  # output dim of decoder = original input dim
            n_layers=decoder_n_layers,
            n_hidden=decoder_n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            use_activation=not use_linear_decoder,
            **_extra_decoder_kwargs
        )

        """model priors p(xz), p_theta(zd|d), p_theta(zy|y)"""

        # calling forward on those encoders directly returns a distribution dist and sample latent
        self.prior_zd_d_encoder = Encoder(
            n_input=n_batch,
            n_output=n_latent_d,
            n_layers=priors_n_layers,
            n_hidden=priors_n_hidden,
            dropout_rate=self.dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
            **_extra_encoder_kwargs
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
            **_extra_encoder_kwargs
        )

        """library model q(l|x)"""
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,  # TODO: add params or kwargs param for such stuff to __init__
            n_hidden=128,  # TODO: add params or kwargs param for such stuff to __init__
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
            **_extra_encoder_kwargs,
        )

        """variational posteriors q_phi(zx|x), q_phi(zd|x), q_phi(zy|x)"""
        self.posterior_zx_x_encoder = Encoder(
            n_input=n_input,
            n_output=n_latent_x,
            n_layers=posterior_n_layers,
            n_hidden=posterior_n_hidden,
            distribution=self.latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
            **_extra_encoder_kwargs
        )

        self.posterior_zd_x_encoder = Encoder(
            n_input=n_input,
            n_output=n_latent_d,
            n_layers=encoder_zd_x_layers,
            n_hidden=encoder_zd_x_hidden,
            distribution=self.latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
            **_extra_encoder_kwargs
        )

        self.posterior_zy_x_encoder = Encoder(
            n_input=n_input,
            n_output=n_latent_y,
            n_layers=posterior_n_layers,
            n_hidden=posterior_n_hidden,
            distribution=self.latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
            **_extra_encoder_kwargs
        )

        """auxiliary tasks q_w(d|zd), q_w(y|zy)"""
        self.aux_d_zd_enc = torch.nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_latent_d, n_batch)
        )

        self.aux_y_enc = torch.nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_latent_y, n_labels)
        )

    def _get_inference_input(self, tensors: dict[str, torch.Tensor], **kwargs):
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        label_index = tensors[REGISTRY_KEYS.LABELS_KEY]

        # train sample
        x = tensors[REGISTRY_KEYS.X_KEY]

        return {'x': x, 'batch_index': batch_index, "label_index": label_index}

    def _get_generative_input(
        self, tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        **kwargs
    ):
        zd = inference_outputs['zd_x']
        zy = inference_outputs['zy_x']
        zx = inference_outputs['zx_x']
        library = inference_outputs['library']
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]

        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = (
            torch.log(tensors[size_factor_key]) if size_factor_key in tensors.keys() else None
        )

        input_dict = {
            'zd_x': zd,
            'zy_x': zy,
            'zx_x': zx,
            'library': library,
            'batch_index': batch_index,
            'y': y,
            'size_factor': size_factor
        }
        return input_dict

    @auto_move_data
    def inference(
        self,
        x,
        batch_index,
        n_samples: int = 1,
        **kwargs
    ) -> dict[str, torch.Tensor | torch.distributions.Distribution]:
        """
        zd_x, zx_x, zy_x, q_zd_x, q_zx_x, q_zy_x, library
        """

        x_ = x
        # library
        library = None
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # tuples of posterior distribution and drawn latent variable
        q_zd_x, zd_x = self.posterior_zd_x_encoder(x_)
        q_zx_x, zx_x = self.posterior_zx_x_encoder(x_)
        q_zy_x, zy_x = self.posterior_zy_x_encoder(x_)

        ql = None
        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(x_, batch_index)
            library = library_encoded

        if n_samples > 1:
            utran_zd = q_zd_x.rsample((n_samples,))
            utran_zx = q_zx_x.rsample((n_samples,))
            utran_zy = q_zy_x.rsample((n_samples,))
            zd_x = self.posterior_zd_x_encoder.z_transformation(utran_zd)
            zx_x = self.posterior_zx_x_encoder.z_transformation(utran_zx)
            zy_x = self.posterior_zy_x_encoder.z_transformation(utran_zy)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.rsample((n_samples,))

        outputs = {'zd_x': zd_x, 'zx_x': zx_x, 'zy_x': zy_x, 'q_zd_x': q_zd_x, 'q_zx_x': q_zx_x, 'q_zy_x': q_zy_x,
                   'library': library}
        return outputs

    @auto_move_data
    def generative(
        self,
        zd_x: torch.Tensor,
        zx_x: torch.Tensor,
        zy_x: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        y: torch.Tensor,
        size_factor=None
    ) -> dict[str, torch.Tensor | torch.distributions.Distribution]:

        if not self.use_size_factor_key:
            size_factor = library

        px_scale, px_r, px_rate, px_dropout = self.reconst_dxy_decoder(
            self.dispersion,
            torch.cat([zd_x, zx_x, zy_x], dim=-1),
            size_factor
        )

        # dispersion
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        # gene likelihood distribution p_theta(x|zd, zx, zy)

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
        p_zd_d, _ = self.prior_zd_d_encoder(one_hot(batch_index, self.n_batch))
        p_zx = Normal(torch.zeros_like(zx_x), torch.ones_like(zx_x))
        p_zy_y, _ = self.prior_zy_y_encoder(one_hot(y, self.n_labels))

        # auxiliary losses
        d_hat = self.aux_d_zd_enc(zd_x)
        y_hat = self.aux_y_enc(zy_x)

        outputs = {'px_recon': px_recon, 'd_hat': d_hat, 'y_hat': y_hat, 'p_zd_d': p_zd_d, 'p_zx': p_zx,
                   'p_zy_y': p_zy_y}

        return outputs

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,  # epsilon
        ce_weight: float = 1.0,
        **kwargs
    ) -> LossOutput:

        """computing the supervised loss (at this point) using
        Inference outputs:
        {'zd_x': zd_x, 'zx_x': zx_x, 'zy_x': zy_x, 'q_zd_x': q_zd_x,
        'q_zx_x': q_zx_x, 'q_zy_x': q_zy_x, 'library': library}
        Generative outputs:
        {'px_recon': px_recon, 'd_hat': d_hat, 'y_hat': y_hat,
        'p_zd_d': p_zd_d, 'p_zx': p_zx, 'p_zy_y': p_zy_y}
        """

        x = tensors[REGISTRY_KEYS.X_KEY]
        d = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]

        # monte carlo approximation of reconstruction loss, uses a subset of the four dimensional monte-carlo sampling
        # required when solving the model reconstruction loss with monte carlo approximations

        reconst_loss = -generative_outputs["px_recon"].log_prob(x).sum(-1)

        # kl_zx = torch.sum(inference_outputs["q_zx_x"].log_prob(zx_x) - generative_outputs["p_zx"].log_prob(zx_x),
        #                   dim=-1)
        kl_zx = kl(
            inference_outputs["q_zx_x"],
            generative_outputs["p_zx"]
        ).sum(dim=1)
        # kl_zy = torch.sum(inference_outputs["q_zy_x"].log_prob(zy_x) - generative_outputs["p_zy_y"].log_prob(zy_x),
        #                   dim=-1)
        kl_zy = kl(
            inference_outputs["q_zy_x"],
            generative_outputs["p_zy_y"]
        ).sum(dim=1)
        # kl_zd = torch.sum(inference_outputs["q_zd_x"].log_prob(zd_x) - generative_outputs["p_zd_d"].log_prob(zd_x),
        #                   dim=-1)
        kl_zd = kl(
            inference_outputs["q_zd_x"],
            generative_outputs["p_zd_d"]
        ).sum(dim=1)

        if not self.use_observed_lib_size:
            kl_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_l = torch.tensor(0.0, device=x.device)

        kl_local_for_warmup = self.beta_x * kl_zx + self.beta_y * kl_zy + self.beta_d * kl_zd
        kl_local_no_warmup = kl_l
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        # auxiliary losses
        d_pred: torch.Tensor

        # weights for cross entropy
        if self._ce_weights_y is None:
            warnings.warn(
                "No weights initialized for cross entropy of DIVA model.\n"
                "Use `.init_ce_weights(adata)` to initialize them.")

        aux_d = F.cross_entropy(generative_outputs["d_hat"], d.view(-1, ), )
        aux_y = F.cross_entropy(generative_outputs["y_hat"], y.view(-1, ),
                                weight=torch.tensor(self._ce_weights_y, device=x.device, dtype=x.dtype))
        aux_loss = self.alpha_d * aux_d + self.alpha_y * aux_y

        loss = torch.mean(reconst_loss + weighted_kl_local) + ce_weight * aux_loss

        kl_local = {
            "kl_divergence_l": kl_l,
            "kl_divergence_zd": kl_zd,
            "kl_divergence_zx": kl_zx,
            "kl_divergence_zy": kl_zy
        }

        return LossOutput(loss, reconst_loss, kl_local, extra_metrics={
            'aux_d': aux_d,
            'aux_y': aux_y
        })

    def sample(self, *args, **kwargs):
        # not really needed for our experiments
        raise NotImplementedError

    @auto_move_data
    def predict(self, tensors) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param tensors: Input tensors of batch as provided by dataloader
        :return: Tuple of (y_true, y_pred) where y_pred is the prediction and y_true the ground truth
        """
        x = tensors[REGISTRY_KEYS.X_KEY]

        if self.log_variational:
            x_ = torch.log(1 + x)
        else:
            x_ = x

        _, zy_x = self.posterior_zy_x_encoder(x_)
        _, y_pred = self.aux_y_enc(zy_x).max(dim=1)
        y_true = tensors[REGISTRY_KEYS.LABELS_KEY]
        return y_true, y_pred

    def init_ce_weight_y(self, adata: AnnData, indices, label_key: str):
        # BEWARE!!! Validation loss will use training weighting for CE!
        # print(cat_d.min(),cat_d.max())
        # print(np.array(range(self.n_batch)))
        cat_y = adata[indices].obs[label_key].cat.codes
        self._ce_weights_y = compute_class_weight('balanced', classes=np.array(range(self.n_labels)), y=cat_y)

    def get_latent(
        self,
        tensors,
        give_mean: bool = True,
        mc_samples: int = 5000,
    ):
        inference_inputs = self._get_inference_input(tensors)
        outputs = self.inference(**inference_inputs)
        q_zd_x = outputs["q_zd_x"]
        q_zx_x = outputs["q_zx_x"]
        q_zy_x = outputs["q_zy_x"]

        if give_mean:
            if self.module.latent_distribution == "ln":
                samples_zd_x = q_zd_x.sample([mc_samples])
                samples_zx_x = q_zx_x.sample([mc_samples])
                samples_zy_x = q_zy_x.sample([mc_samples])

                zd_x = torch.nn.functional.softmax(samples_zd_x, dim=-1)
                zx_x = torch.nn.functional.softmax(samples_zx_x, dim=-1)
                zy_x = torch.nn.functional.softmax(samples_zy_x, dim=-1)

                zd_x = zd_x.mean(dim=0)
                zx_x = zx_x.mean(dim=0)
                zy_x = zy_x.mean(dim=0)

            else:
                zd_x = q_zd_x.loc
                zx_x = q_zx_x.loc
                zy_x = q_zy_x.loc
        else:
            zd_x = outputs["zd_x"]
            zx_x = outputs["zx_x"]
            zy_x = outputs["zy_x"]

        return [torch.cat([zd_x.cpu(), zx_x.cpu(), zy_x.cpu()], dim=1)]
