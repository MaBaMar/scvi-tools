"""DIVA implementation supervised"""
from __future__ import annotations

import logging
import sys
import warnings
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from anndata import AnnData
from matplotlib.patheffects import Normal
from scvi import REGISTRY_KEYS
from scvi._types import Tunable
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial, Poisson
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder, one_hot, MeanOnlyEncoder, RQMDecoder, CholeskyCovEncoder, DiagMultEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.distributions import Normal, kl_divergence as kl, MultivariateNormal, register_kl

logger = logging.Logger(__name__)


@register_kl(Normal, MultivariateNormal)
def _kl_normal_multivariate_normal(p: Normal, q):
    p = MultivariateNormal(p.loc, torch.diag_embed(p.stddev ** 2))
    return kl(p, q)


# TODO: support minification
class DIVA(BaseModuleClass):
    """base DIVA implementation, slight adaptation of model from original paper"""

    _ce_weights_y = None
    _kl_weights_y = None
    _kl_weights_d = None
    _unsupervised: bool = False

    def __init__(
        self,
        n_input: int,
        n_batch: int,
        n_labels: int,
        n_latent_d: int = 4,
        n_latent_y: int = 10,
        beta_d: float = 1,
        beta_y: float = 1,
        alpha_d: float = 1000,
        alpha_y: float = 1500,
        priors_n_hidden: int = 32,
        priors_n_layers: int = 1,
        prior_variance_d: float = None,
        prior_variance_y: float = None,
        posterior_n_hidden: int = 128,
        posterior_n_layers: int = 1,
        decoder_n_hidden: int = 128,
        decoder_n_layers: int = 1,
        dropout_rate: float = 0.1,
        lib_encoder_n_hidden: int = 128,
        lib_encoder_n_layers: int = 1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: Tunable[bool] = True,  # TODO: fix
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_linear_batch_classifier: bool = False,
        use_linear_label_classifier: bool = False,
        use_prior_based_predictions: bool = False,
        use_size_factor_key: bool = False,
        use_observed_lib_size: Tunable[bool] = True,
        use_learnable_priors: Tunable[bool] = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        extra_encoder_kwargs: Optional[dict] = None,
        extra_decoder_kwargs: Optional[dict] = None,
        **kwargs
    ):

        """
        Module for DIVA based models. Supports various sub-variants and customizations. Inspired by the original DIVA
        implementation, adapted and evolved for the single cell RNA-seq case

        Parameters
        ----------
        n_input
            Number of input features, i.e. different genes per sample
        n_batch
            Number of batches in the dataset upon which `setup_adata` was called
        n_labels
            Number of different cell-types/labels in the dataset upon which `setup_adata` was called
        n_latent_d
            Dimension for batch latent space
        n_latent_y
            Dimension of cell-type latent space. If 0, the latent space is deactivated. The latent space can be
            deactivated to use the model for batch-integration tasks, effectively making the model unsupervised in terms
            of cell-type labels.
        beta_d
            KL weight :math:`\beta_{d}` used to weight the KL divergence term of the batch latent space. If learnable
            priors are used, this parameter drives separation of batch clusters and helps avoid capturing non-batch
            related information in the batch latent space. Also serves as a regularizer.
        beta_y
            KL weight :math:`\beta_{y}` used to weight the KL divergence term of the cell-type latent space. If
            learnable priors are used, this parameter drives separation of cell-type clusters and helps avoid capturing
            non-cell-type related information in the cell-type latent space. Also serves as a regularizer.
        alpha_d
            Classification weight :math:`\alpha_{d}` used to weight CE loss of batch classifier (if used). To deactivate
             the classifier, set this value to 0 or set `min_classifier_weight`=0 and `max_classifier_weight`=0 in
            `plan_kwargs` of the training plan. The latter deactivates both cell-type and batch classifiers.
        alpha_y
            Classification weight :math:`\alpha_{y}` used to weight CE loss of cell-type classifier (if used). To
            deactivate the classifier, set this value to 0 or set `min_classifier_weight`=0 and
            `max_classifier_weight`=0 in `plan_kwargs` of the training plan. The latter deactivates both cell-type and
            batch classifiers.
        priors_n_hidden
            Number of hidden units in the prior encoders used for the learnable priors. If `use_learnable_priors` is
            `False`, this value will be ignored.
        priors_n_layers
            Number of hidden layers in the prior encoders used for the learnable priors. If `use_learnable_priors` is
            `False`, this value will be ignored.
        prior_variance_d
            If this is not `None`, the model will use the given constant variance for all batch priors along all
            dimensions instead of learning it. If `use_learnable_priors` is `False`, this value will be ignored.
        prior_variance_y
            If this is not `None`, the model will use the given constant variance for all cell-type priors along all
            dimensions instead of learning it. If `use_learnable_priors` is `False`, this value will be ignored.
        arches_batch_extension_size
            This parameter can be used to internally extend the input size and is used for training with scARCHES. The
            parameter should never be set manually. Instead, the corresponding modules will set it automatically.
        posterior_n_hidden
            Number of hidden units in the posterior encoders. All posteriors use the same amount of hidden units to
            reduce hyperparameter count.
        posterior_n_layers
            Number of hidden layers in the posterior encoders. All posteriors use the same amount of hidden layers to
            reduce hyperparameter count.
        decoder_n_hidden
            Number of hidden units in the SCVI decoder.
        decoder_n_layers
            Number of hidden layers in the SCVI decoder.
        dropout_rate
            Dropout rate used for the learnable priors. If `use_learnable_priors` is `False`, this value will be
            ignored.
        dispersion
            One of the following:

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
            Model has only been tested with `gene` so far. Option exists for API compatibility. Use at your own risk.
        log_variational
            Log(data+1) prior to encoding for numerical stability. Not normalization.
        gene_likelihood
            One of:

            * ``'nb'`` - Negative binomial distribution
            * ``'zinb'`` - Zero-inflated negative binomial distribution
            * ``'poisson'`` - Poisson distribution
        latent_distribution
            One of

            * ``'normal'`` - Isotropic normal
            * ``'ln'`` - Logistic normal with normal params N(0, 1)
        use_batch_norm
            Whether to use batch norm in layers.
        use_layer_norm
            Whether to use layer norm in layers.
        use_linear_batch_classifier
            Whether to make the batch classifier linear. This parameter is ignored if the batch classifier is
            deactivated.
        use_linear_label_classifier
            Whether to make the batch cell-type linear. This parameter is ignored if the batch classifier is
            deactivated.
        use_size_factor_key
            Use size_factor AnnDataField defined by the user as scaling factor in mean of conditional distribution.
            Takes priority over `use_observed_lib_size`.
        use_observed_lib_size
            Use observed library size for RNA as scaling factor in mean of conditional distribution
        use_learnable_priors
            If `False`, the priors used for batch and cell-type latent space correspond to normal distributions and
            are no longer learnable.
        library_log_means
            1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
            not using observed library size.
        library_log_vars
            1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
            not using observed library size.
        extra_encoder_kwargs
            Additional keyword arguments to pass to `Encoder` classes.
        extra_decoder_kwargs
            Additional keyword arguments to pass to `DecoderSCVI` or `LinearDecoderSCVI` classes.
        """
        super().__init__()

        if n_latent_y <= 0:
            print("\ny latent dimension is <= 0. Disabling y latent space and automatically switching to unsupervised",
                  "training mode. If you want to use the y latent space consider increasing",
                  "n_latent_y to an int value > 0", file=sys.stderr)
            n_latent_y = 0
            self._unsupervised = True

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
        self.alpha_d = alpha_d

        # if latent spaces are used, but the classifiers deactivated, the classifiers are removed from inference to
        # speed up the model
        self._use_batch_classifier = alpha_d != 0
        self._use_celltype_classifier = alpha_y != 0 and not self._unsupervised and not use_prior_based_predictions

        logger.info(f"Using batch classifier:  {self._use_batch_classifier}")
        logger.info(f"Using celltype classifier:  {self._use_celltype_classifier}")

        self.use_learnable_priors = use_learnable_priors

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
        self.reconstruction_dxy_decoder = RQMDecoder(
            n_input_y=n_latent_y,
            n_input_d=n_latent_d,
            n_output=n_input,  # output dim of decoder = original input dim
            n_layers=decoder_n_layers,
            n_hidden=decoder_n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            inject_covariates=False,
            **_extra_decoder_kwargs
        )

        """model priors p(xz), p_theta(zd|d), p_theta(zy|y)"""

        # calling forward on those encoders directly returns a distribution dist and sample latent
        if use_learnable_priors:
            if prior_variance_d is None:
                self.prior_zd_d_encoder = Encoder(
                    n_input=n_batch,
                    n_output=n_latent_d,
                    n_layers=priors_n_layers,
                    n_hidden=priors_n_hidden,
                    dropout_rate=self.dropout_rate,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    return_dist=True,
                    inject_covariates=False,
                    **_extra_encoder_kwargs
                )
            else:
                self.prior_zd_d_encoder = MeanOnlyEncoder(
                    n_input=n_batch,
                    n_output=n_latent_d,
                    n_layers=priors_n_layers,
                    n_hidden=priors_n_hidden,
                    dropout_rate=self.dropout_rate,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    return_dist=True,
                    var=prior_variance_d,
                    inject_covariates=False,
                    **_extra_encoder_kwargs
                )

            # deactivate the y latent space for unsupervised mode
            if not self._unsupervised:
                if prior_variance_y is None:
                    self.prior_zy_y_encoder = CholeskyCovEncoder(
                        n_input=n_labels - 1,
                        n_output=n_latent_y,
                        n_layers=priors_n_layers,
                        n_hidden=priors_n_hidden,
                        dropout_rate=self.dropout_rate,
                        use_batch_norm=False,
                        use_layer_norm=False,
                        return_dist=True,
                        inject_covariates=False,
                        **_extra_encoder_kwargs
                    )
                else:
                    self.prior_zy_y_encoder = MeanOnlyEncoder(
                        n_input=n_labels - 1,
                        n_output=n_latent_y,
                        n_layers=priors_n_layers,
                        n_hidden=priors_n_hidden,
                        dropout_rate=self.dropout_rate,
                        use_batch_norm=False,
                        use_layer_norm=False,
                        return_dist=True,
                        var=prior_variance_y,
                        inject_covariates=False,
                        **_extra_encoder_kwargs
                    )

        """library model q(l|x)"""
        # l encoder goes from n_input-dimensional data to 1-d library size
        if not use_observed_lib_size:
            self.l_encoder = Encoder(
                n_input,
                1,
                n_layers=lib_encoder_n_layers,
                n_hidden=lib_encoder_n_hidden,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm_encoder,
                use_layer_norm=use_layer_norm_encoder,
                return_dist=True,
                inject_covariates=False,
                **_extra_encoder_kwargs,
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
            inject_covariates=False,
            **_extra_encoder_kwargs
        )

        # unsupervised mode has no need for the y latent posterior and classifier
        if not self._unsupervised:
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
                inject_covariates=False,
                **_extra_encoder_kwargs
            )

            """auxiliary task q_w(y|zy), i.e. y-latent classifier -> only used if the latent classifier is actually
            activated (this is never the case in unsupervised mode)"""
            if self._use_celltype_classifier:
                self.aux_y_zy_enc = torch.nn.Sequential(
                    *[] if use_linear_label_classifier else [nn.ReLU()],
                    nn.Linear(n_latent_y, n_labels - 1)
                )

        """auxiliary task q_w(d|zd)"""
        if self._use_batch_classifier:
            self.aux_d_zd_enc = torch.nn.Sequential(
                *[] if use_linear_batch_classifier else [nn.ReLU()],
                nn.Linear(n_latent_d, n_batch)
            )

    def _get_inference_input(self, tensors: dict[str, torch.Tensor], **kwargs):
        return {'x': tensors[REGISTRY_KEYS.X_KEY], 'd': tensors[REGISTRY_KEYS.BATCH_KEY]}

    def _get_generative_input(
        self, tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        **kwargs
    ):
        zd = inference_outputs['zd_x']
        zy = inference_outputs['zy_x']
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
            'library': library,
            'd': batch_index,
            'y': y,
            'size_factor': size_factor
        }
        return input_dict

    @auto_move_data
    def inference(
        self,
        x,
        d,
        n_samples: int = 1,
        **kwargs
    ) -> dict[str, torch.Tensor | torch.distributions.Distribution]:

        # library size
        library = None
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x = torch.log(1 + x)

        # tuples of posterior distribution and drawn latent variable
        q_zd_x, zd_x = self.posterior_zd_x_encoder(x)
        if not self._unsupervised:
            q_zy_x, zy_x = self.posterior_zy_x_encoder(x)
        else:
            q_zy_x, zy_x = (None, None)

        ql = None
        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(x, d)
            library = library_encoded

        if n_samples > 1:
            utran_zd = q_zd_x.rsample((n_samples,))
            zd_x = self.posterior_zd_x_encoder.z_transformation(utran_zd)
            if not self._unsupervised:
                utran_zy = q_zy_x.rsample((n_samples,))
                zy_x = self.posterior_zy_x_encoder.z_transformation(utran_zy)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.rsample((n_samples,))

        outputs = {'zd_x': zd_x, 'zy_x': zy_x, 'q_zd_x': q_zd_x, 'q_zy_x': q_zy_x, 'library': library}
        return outputs

    @auto_move_data
    def generative(
        self,
        zd_x: torch.Tensor,
        zy_x: torch.Tensor,
        library: torch.Tensor,
        d: torch.Tensor,
        y: torch.Tensor,
        size_factor=None
    ) -> dict[str, torch.Tensor | torch.distributions.Distribution]:

        if not self.use_size_factor_key:
            size_factor = library

        px_scale, px_r, px_rate, px_dropout = self.reconstruction_dxy_decoder(
            self.dispersion,
            zy_x,
            zd_x,
            size_factor
        )

        # dispersion
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(d, self.n_batch), self.px_r)
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
        p_zy_y = None
        if self.use_learnable_priors:
            p_zd_d, _ = self.prior_zd_d_encoder(one_hot(d, self.n_batch))
            if not self._unsupervised:
                p_zy_y, _ = self.prior_zy_y_encoder(one_hot(y, self.n_labels - 1))
        else:
            p_zd_d = Normal(torch.zeros_like(zd_x), torch.ones_like(zd_x))
            if not self._unsupervised:
                p_zy_y = Normal(torch.zeros_like(zy_x), torch.ones_like(zy_x))

        # auxiliary losses
        d_hat = self.aux_d_zd_enc(zd_x) if self._use_batch_classifier else None
        # """
        if self._use_celltype_classifier:
            y_hat = self._pred_cls(zy_x)
        elif not self._unsupervised:
            y_hat = self._pred_prior(zy_x)
        else:
            y_hat = None

        outputs = {'px_recon': px_recon, 'd_hat': d_hat, 'y_hat': y_hat, 'p_zd_d': p_zd_d, 'p_zy_y': p_zy_y}

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

        neg_reconstruction_loss = -generative_outputs["px_recon"].log_prob(x).sum(-1)

        kl_zd = kl(
            inference_outputs["q_zd_x"],
            generative_outputs["p_zd_d"]
        ).sum(dim=-1)

        if not self.use_observed_lib_size:
            kl_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_l = torch.tensor(0.0, device=x.device)

        # KL loss
        if self._kl_weights_d is None:
            warnings.warn(
                "No weights initialized for balanced KL of DIVA model.\n"
                "Use `.init_kl_weights` to initialize them.")

        kl_local_for_warmup = self.beta_d * kl_zd * self._kl_weights_d.to(self.device)[d].flatten()

        if not self._unsupervised:
            kl_zy = kl(
                inference_outputs["q_zy_x"],
                generative_outputs["p_zy_y"]
            )

            kl_local_for_warmup += self.beta_y * kl_zy * self._kl_weights_y.to(self.device)[y].flatten()
        else:
            kl_zy = 0

        aux_loss = 0
        extra_metrics = {}

        if self._use_batch_classifier:
            # auxiliary losses
            aux_d = F.cross_entropy(generative_outputs["d_hat"], d.flatten())
            aux_loss += self.alpha_d * aux_d
            extra_metrics["aux_d"] = aux_d

        if not self._unsupervised:
            aux_y = F.cross_entropy(generative_outputs["y_hat"], y.flatten(), weight=self._ce_weights_y.to(self.device))
            aux_loss += self.alpha_y * aux_y
            extra_metrics["aux_y"] = aux_y

        kl_local_no_warmup = kl_l
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(neg_reconstruction_loss + weighted_kl_local) + ce_weight * aux_loss

        kl_local = {
            "kl_divergence_l": kl_l,
            "kl_divergence_zd": kl_zd,
            "kl_divergence_zy": kl_zy if not self._unsupervised else 0
        }

        extra_metrics.update({'kl_l': kl_l.mean(), 'kl_zd': kl_zd.mean(), 'kl_zy': kl_zy.mean()})

        return LossOutput(loss, neg_reconstruction_loss, kl_local, extra_metrics=extra_metrics)

    def sample(self, *args, **kwargs):
        # not really needed for our experiments
        raise NotImplementedError

    @auto_move_data
    @torch.inference_mode
    def predict(self, tensors, mode: Literal['prior_based', 'internal_classifier'], use_mean_as_sample=False, n_samples=100):
        """
        Uses the model's internal prediction mechanisms to make predictions on query data. Do not use this method if you
        want to use downstream classifiers instead.

        Parameters
        ----------
        tensors
            The query data
        mode
            One of 'prior_based' or 'internal_classifier'. 'prior_based' uses the learned priors for predictions
            (if available) while 'internal_classifier' uses the internally trained cell-type classifier
        use_mean_as_sample
            If true, uses the mean of the posterior normal distribution as sample instead of drawing from the posterior
            distribution. This reduces variance in predictions and is generally preferable in benchmarking settings
        n_samples
            Determines the number of Monte Carlo samples to draw (i.e. draws of zy_x). If `use_mean_as_sample` is true, this argument is
            ignored.

        Returns
        -------
        torch.Tensor
            The predicted cell type labels for the given data
        """

        if not mode in (md_tmp := ['prior_based', 'internal_classifier']):
            raise ValueError(f"mode must be in {md_tmp}")

        if self._unsupervised:
            raise NotImplementedError("Unsupervised prediction not supported. Please use an external classifier")

        x = tensors[REGISTRY_KEYS.X_KEY]
        if self.log_variational:
            x = torch.log(1 + x)

        # use the mode to determine the prediction function
        match mode:
            case 'prior_based':
                predictor_fn = self._pred_prior
            case 'internal_classifier':
                if not self._use_celltype_classifier:
                    warnings.warn('internal_classifier mode requires `alpha_y > 0`. Using prior_based mode instead.',
                                  category=RuntimeWarning)
                    predictor_fn = self._pred_prior
                else:
                    predictor_fn = self._pred_cls
            case _:
                raise ValueError("unsupported mode")

        dist, _ = self.posterior_zy_x_encoder(x)

        if use_mean_as_sample:
            probs = predictor_fn(dist.mean)
        else:
            pred_avgs = torch.zeros((n_samples, x.shape[0], self.n_labels - 1))
            for i in range(n_samples):
                y_pred = F.softmax(predictor_fn(dist.sample()), dim=-1)
                pred_avgs[i] = y_pred
            probs = torch.mean(pred_avgs, dim=0)
        return torch.argmax(probs, dim=1)

    def get_prediction_uncertainty(self, tensors, mode: Literal['prior_based', 'internal_classifier']):

        # auto move did not work for directory
        for key in tensors.keys():
            tensors[key] = tensors[key].to(self.device)

        if not mode in (md_tmp := ['prior_based', 'internal_classifier']):
            raise ValueError(f"mode must be in {md_tmp}")

        if self._unsupervised:
            raise NotImplementedError("Unsupervised prediction not supported. Please use an external classifier")

        x = tensors[REGISTRY_KEYS.X_KEY]
        if self.log_variational:
            x = torch.log(1 + x)

        # use the mode to determine the prediction function
        match mode:
            case 'prior_based':
                predictor_fn = self._pred_prior
            case 'internal_classifier':
                if not self._use_celltype_classifier:
                    warnings.warn('internal_classifier mode requires `alpha_y > 0`. Using prior_based mode instead.',
                                  category=RuntimeWarning)
                    predictor_fn = self._pred_prior
                else:
                    predictor_fn = self._pred_cls
            case _:
                raise ValueError("unsupported mode")

        dist, _ = self.posterior_zy_x_encoder(x)

        probs = F.softmax(predictor_fn(dist.mean), dim=-1)
        y = tensors[REGISTRY_KEYS.LABELS_KEY].flatten()
        return probs.gather(1, y.unsqueeze(1)).squeeze(1).cpu().numpy(), y.cpu().numpy()

    def _pred_prior(self, zy_x: torch.Tensor) -> torch.Tensor:
        encodings = torch.eye(self.n_labels - 1, device=self.device)
        p_zy_y: torch.distributions.MultivariateNormal
        p_zy_y, _ = self.prior_zy_y_encoder(encodings)
        return p_zy_y.log_prob(zy_x.unsqueeze(1))

    def _pred_cls(self, zy_x: torch.Tensor) -> torch.Tensor:
        return self.aux_y_zy_enc(zy_x)

    def init_ce_weight_y(self, adata: AnnData, indices, label_key: str):
        """
        Initializes the cross entropy weights for the training process. All data occuring in the training set will be weighted accordingly.
        Data not present in the proviced adata will be ignored. Also, the adata object may not include any unlabeled samples as the reference
        building process is purely supervised.
        # BEWARE!!! Validation loss will currently use training weighting for CE!
        :param adata: The training adata to fit to
        :param indices: If the adata contains more than just the training samples, these indices should be used to indicate which samples should be used
        :param label_key: The label key
        :return:
        """
        cat_y = adata.obs[label_key].cat.codes[indices]
        data_index_space = np.unique(cat_y)
        model_index_space = np.array(range(self.n_labels - 1))

        if len(t := np.setdiff1d(data_index_space, model_index_space)) > 0:
            raise ValueError('Data contains classes with indices {}. Those are not supported by the used model instance. Make sure you only'
                             'use data that was set up for the model instance.'.format(t))

        weight_arr = np.zeros(self.n_labels - 1)
        weight_arr[data_index_space] = compute_class_weight('balanced', classes=data_index_space, y=cat_y)

        self._ce_weights_y = torch.tensor(weight_arr, device=self.device, dtype=torch.float)

    def init_kl_weights(self, adata: AnnData, indices, label_key: str, batch_key: str):
        # TODO: needs some rework!
        cat_y = adata.obs[label_key].cat.codes[indices]
        cat_d = adata.obs[batch_key].cat.codes[indices]

        cat_d_holdout = np.setdiff1d(np.unique(adata.obs[batch_key].cat.codes),
                                     np.unique(adata.obs[batch_key].cat.codes[indices]))
        cat_y_holdout = np.setdiff1d(np.unique(adata.obs[label_key].cat.codes),
                                     np.unique(adata.obs[label_key].cat.codes[indices]))

        """Note: We might have batches only present in validation set. This needs to be taken into consideration here.
        They'll get very high training weights, but since we do not use those batches during training, that does
        not matter. However, the high values might corrupt validation loss. Therefore, we set such batches to 1"""

        """use # samples / (1 + # classes * # bincount) to allow calculation for empty batches (here 1+ in divisor)"""
        cat_d_corrected = np.append(cat_d, range(self.n_batch))
        cat_y_corrected = np.append(cat_y, range(self.n_labels - 1))

        self._kl_weights_y = torch.from_numpy(
            compute_class_weight('balanced', classes=np.array(range(self.n_labels - 1)), y=cat_y_corrected))
        self._kl_weights_d = torch.from_numpy(compute_class_weight('balanced', classes=np.array(range(self.n_batch)), y=cat_d_corrected))

        self._kl_weights_y[cat_y_holdout] = 1
        self._kl_weights_d[cat_d_holdout] = 1

    @torch.inference_mode()
    def sample_log_likelihood(self, tensors, n_samples: int):
        x = tensors[REGISTRY_KEYS.X_KEY]
        d = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]

        raise NotImplementedError('This method is not fully implemented.')

        if self.log_variational:
            x = torch.log(1 + x)
        llh_sum = 0
        for i in range(n_samples):
            inference_input = self._get_inference_input(tensors)
            self._inference_input = inference_input
            # HANDLE SIZE FACTOR
            size_factor = ...  # TODO: complete
            zd_x = self.posterior_zd_x_encoder(x)
            zy_x = self.posterior_zy_x_encoder(x)
            self.reconstruction_dxy_decoder(
                self.dispersion,
                zy_x,
                zd_x,
                size_factor
            )
