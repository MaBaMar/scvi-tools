from __future__ import annotations

import logging
import warnings
from typing import Literal, Optional, Sequence, Union

import numpy as np
import torch
from anndata import AnnData
from lightning import LightningDataModule
from scvi import REGISTRY_KEYS, settings
from scvi._types import Tunable
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField, CategoricalObsField
from scvi.dataloaders._data_splitting import DefaultDataSplitter
from scvi.model._utils import _init_library_size, get_max_epochs_heuristic, use_distributed_sampler
from scvi.model.base import RNASeqMixin, VAEMixin, BaseModelClass, UnsupervisedTrainingMixin
from scvi.module import DIVA
from scvi.train._trainingplans import ReferenceQueryPlan
from scvi.utils import setup_anndata_dsp

logger = logging.getLogger(__name__)


class DiSCVI(RNASeqMixin, VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Domain independent single cell variational inference
    """

    _module_cls = DIVA
    _batch_key = None
    _label_key = None

    def __init__(
        self,
        adata: AnnData | None = None,
        n_latent_d: int = 3,
        n_latent_x: int = 4,
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
        self._training_plan_cls = ReferenceQueryPlan
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

        cls._batch_key = batch_key
        cls._label_key = labels_key

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
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
            latent += self.module.get_latent(tensors, give_mean=give_mean, mc_samples=mc_samples)

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
        :returns: (y_true, y_pred)
        """
        y_pred = []
        y_true = []
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        for tensors in scdl:
            model_pred = self.module.predict(tensors)
            y_true.append(model_pred[0].cpu())
            y_pred.append(model_pred[1].cpu())

        return torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0)

    def train(
        self,
        max_epochs: int | None = None,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        batch_size: Tunable[int] = 128,
        early_stopping: bool = False,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        data_module: LightningDataModule | None = None,
        **trainer_kwargs,
    ):
        """Train the model.

                Parameters
                ----------
                max_epochs
                    The maximum number of epochs to train the model. The actual number of epochs may be
                    less if early stopping is enabled. If ``None``, defaults to a heuristic based on
                    :func:`~scvi.model.get_max_epochs_heuristic`. Must be passed in if ``data_module`` is
                    passed in, and it does not have an ``n_obs`` attribute.
                %(param_accelerator)s
                %(param_devices)s
                train_size
                    Size of training set in the range ``[0.0, 1.0]``. Passed into
                    :class:`~scvi.dataloaders.DataSplitter`. Not used if ``data_module`` is passed in.
                validation_size
                    Size of the test set. If ``None``, defaults to ``1 - train_size``. If
                    ``train_size + validation_size < 1``, the remaining cells belong to a test set. Passed
                    into :class:`~scvi.dataloaders.DataSplitter`. Not used if ``data_module`` is passed in.
                shuffle_set_split
                    Whether to shuffle indices before splitting. If ``False``, the val, train, and test set
                    are split in the sequential order of the data according to ``validation_size`` and
                    ``train_size`` percentages. Passed into :class:`~scvi.dataloaders.DataSplitter`. Not
                    used if ``data_module`` is passed in.
                load_sparse_tensor
                    ``EXPERIMENTAL`` If ``True``, loads data with sparse CSR or CSC layout as a
                    :class:`~torch.Tensor` with the same layout. Can lead to speedups in data transfers to
                    GPUs, depending on the sparsity of the data. Passed into
                    :class:`~scvi.dataloaders.DataSplitter`. Not used if ``data_module`` is passed in.
                batch_size
                    Minibatch size to use during training. Passed into
                    :class:`~scvi.dataloaders.DataSplitter`. Not used if ``data_module`` is passed in.
                early_stopping
                    Perform early stopping. Additional arguments can be passed in through ``**kwargs``.
                    See :class:`~scvi.train.Trainer` for further options.
                datasplitter_kwargs
                    Additional keyword arguments passed into :class:`~scvi.dataloaders.DataSplitter`. Values
                    in this argument can be overwritten by arguments directly passed into this method, when
                    appropriate. Not used if ``data_module`` is passed in.
                plan_kwargs
                    Additional keyword arguments passed into :class:`~scvi.train.TrainingPlan`. Values in
                    this argument can be overwritten by arguments directly passed into this method, when
                    appropriate.
                data_module
                    ``EXPERIMENTAL`` A :class:`~lightning.pytorch.core.LightningDataModule` instance to use
                    for training in place of the default :class:`~scvi.dataloaders.DataSplitter`. Can only
                    be passed in if the model was not initialized with :class:`~anndata.AnnData`.
                **kwargs
                   Additional keyword arguments passed into :class:`~scvi.train.Trainer`.
                """
        if data_module is not None and not self._module_init_on_train:
            raise ValueError(
                "Cannot pass in `data_module` if the model was initialized with `adata`."
            )
        elif data_module is None and self._module_init_on_train:
            raise ValueError(
                "If the model was not initialized with `adata`, a `data_module` must be passed in."
            )

        if max_epochs is None:
            if data_module is None:
                max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
            elif hasattr(data_module, "n_obs"):
                max_epochs = get_max_epochs_heuristic(data_module.n_obs)
            else:
                raise ValueError(
                    "If `data_module` does not have `n_obs` attribute, `max_epochs` must be passed "
                    "in."
                )

        if data_module is None:
            datasplitter_kwargs = datasplitter_kwargs or {}
            data_module = self._data_splitter_cls(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                shuffle_set_split=shuffle_set_split,
                distributed_sampler=use_distributed_sampler(trainer_kwargs.get("strategy", None)),
                load_sparse_tensor=load_sparse_tensor,
                **datasplitter_kwargs,
            )
        elif self.module is None:
            self.module = self._module_cls(
                data_module.n_vars,
                n_batch=data_module.n_batch,
                n_labels=getattr(data_module, "n_labels", 1),
                n_continuous_cov=getattr(data_module, "n_continuous_cov", 0),
                n_cats_per_cov=getattr(data_module, "n_cats_per_cov", None),
                **self._module_kwargs,
            )

        plan_kwargs = plan_kwargs or {}
        if not 'n_epochs_kl_warmup' in plan_kwargs:
            plan_kwargs['n_epochs_kl_warmup'] = min(400, max_epochs)
        training_plan = self._training_plan_cls(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_module,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )
        data_module.setup()

        self.module.init_ce_weight_y(self.adata, data_module.train_idx, self._label_key)
        return runner()