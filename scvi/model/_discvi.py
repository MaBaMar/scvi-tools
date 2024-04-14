from __future__ import annotations

import logging
import warnings
from typing import Literal, Optional, Sequence, Union

import numpy as np
import torch
from anndata import AnnData
from scvi.dataloaders._data_splitting import DefaultDataSplitter
from overrides import overrides
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField, CategoricalObsField
from scvi.model._utils import _init_library_size
from scvi.model.base import RNASeqMixin, VAEMixin, BaseModelClass, UnsupervisedTrainingMixin
from scvi.module import DIVA
from scvi.utils import setup_anndata_dsp

logger = logging.getLogger(__name__)


class DiSCVI(RNASeqMixin, VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Domain independent single cell variational inference
    """

    _module_cls = DIVA

    def __init__(
        self,
        adata: AnnData | None = None,
        n_latent_d: int = 3,
        n_latent_x: int = 3,
        n_latent_y: int = 10,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        use_default_data_splitter=False,
        **kwargs
    ):
        """
        :param use_default_data_splitter: If true, you need to pass "default_splitter" argument as datasplitter_kwargs
        when calling train, passing a default datasplitter module
        """
        super().__init__(adata)

        # TODO: potentially overwrite training plan and train runner of _training_mixin.py
        # _training_plan_cls = TrainingPlan
        # _train_runner_cls = TrainRunner

        self._module_kwargs = {
            "n_latent_d": n_latent_d,
            "n_latent_x": n_latent_x,
            "n_latent_y": n_latent_y,
            "dropout_rate": dropout_rate,
            "dispersion": dispersion,
            "gene_likelihood": gene_likelihood,
            "latent_distribution": latent_distribution,
            **kwargs
        }

        self._model_summary_string = (
            "DiSCVI model with the following parameters: \n"
            f"n_latent_d: {n_latent_d}, n_latent_x: {n_latent_x}, n_latent_y: {n_latent_y}, "
            f"dropout_rate: {dropout_rate}, dispersion: {dispersion}, "
            f"gene_likelihood: {gene_likelihood}, latent_distribution: {latent_distribution}, "
        )

        if self._module_init_on_train:
            self.module = None
            warnings.warn(
                "Model was initialized without `adata`. The module will be initialized when "
                "calling `train`. This behavior is experimental and may change in the future.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        else:
            n_batch = self.summary_stats.n_batch
            use_size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
            library_log_means, library_log_vars = None, None
            if not use_size_factor_key:  # TODO: edit if supporting minified
                library_log_means, library_log_vars = _init_library_size(
                    self.adata_manager, n_batch
                )
            self.module = self._module_cls(
                n_input=self.summary_stats.n_vars,
                n_batch=n_batch,
                n_labels=self.summary_stats.n_labels,
                n_latent_d=n_latent_d,
                n_latent_x=n_latent_x,
                n_latent_y=n_latent_y,
                dropout_rate=dropout_rate,
                dispersion=dispersion,
                gene_likelihood=gene_likelihood,
                latent_distribution=latent_distribution,
                use_size_factor_key=use_size_factor_key,
                library_log_means=library_log_means,
                library_log_vars=library_log_vars,
                **kwargs
            )
        self.init_params_ = self._get_init_params(locals())

        if use_default_data_splitter:
            self._data_splitter_cls = DefaultDataSplitter

        # TODO: complete

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        **kwargs
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_size_factor_key)s
        %(param_labels_key)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
        ]

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    @overrides
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
        return_dist: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Return the latent representation for each cell.

        Specifically, returns [ :math:`z_d` , :math:`z_x` , :math:`z_y` ].

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_dist
            Return (mean, variance) of distributions instead of just the mean.
            If `True`, ignores `give_mean` and `mc_samples`. In the case of the latter,
            `mc_samples` is used to compute the mean of a transformed distribution.
            If `return_dist` is true the untransformed mean and variance are returned.

        Returns
        -------
        Low-dimensional representation for each cell or a tuple containing its mean and variance.
        """

        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        latent = []

        if return_dist:
            raise NotImplementedError

        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
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

            latent += [torch.cat([zd_x.cpu(), zx_x.cpu(), zy_x.cpu()], dim=1)]

        return torch.cat(latent).numpy()

    def latent_separated(self, **kwargs):
        """Returns [ :math:`z_d` , :math:`z_x` , :math:`z_y` ]
        Parameters
        ----------
        Returns
        -------
        """
        latent = self.get_latent_representation(**kwargs)
        n_d = self._module_kwargs["n_latent_d"]
        n_x = self._module_kwargs["n_latent_x"] + n_d
        n_y = self._module_kwargs["n_latent_y"] + n_x
        return latent[:, :n_d], latent[:, n_d:n_x], latent[:, n_x:n_y]

    @torch.inference_mode()
    def predict(
        self, adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ):
        """
        :returns: (y, y_hat)
        """
        predictions = []
        labels = []
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        for tensors in scdl:
            model_pred = self.module.predict(tensors)
            predictions.append(model_pred[0].cpu())
            labels.append(model_pred[1].cpu())

        return torch.cat(labels, dim=0), torch.cat(predictions, dim=0)
