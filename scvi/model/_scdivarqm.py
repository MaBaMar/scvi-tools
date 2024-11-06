import warnings
from typing import Optional, Literal, Union, Sequence

import numpy as np
import torch
from anndata import AnnData
from lightning import LightningDataModule
from scvi import REGISTRY_KEYS, settings
from scvi._types import Tunable
from scvi.data import AnnDataManager
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY, _SCVI_VERSION_KEY
from scvi.data.fields import NumericalObsField, LayerField, CategoricalObsField
from scvi.dataloaders._data_splitting import DefaultDataSplitter
from scvi.model._utils import get_max_epochs_heuristic, use_distributed_sampler, parse_device_args
from scvi.model.base import ArchesMixin, UnsupervisedTrainingMixin, RNASeqMixin, VAEMixin, BaseModelClass
from scvi.model.base._archesmixin import _get_loaded_data, _set_params_online_update
from scvi.model.base._utils import _validate_var_names, _initialize_model
from scvi.module._diva_rqm import RQMDiva
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp


class ScDiVarQM(RNASeqMixin, VAEMixin, BaseModelClass, ArchesMixin, UnsupervisedTrainingMixin):
    """
    Streamlined RQM implementation of scDIVA
    """
    _module_cls = RQMDiva
    _batch_key: Optional[str] = None
    _clabel_key: Optional[str] = None

    def __init__(
        self,
        adata: AnnData | None = None,
        n_latent_d: int = 4,
        n_latent_y: int = 10,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        use_default_data_splitter=False,
        **kwargs):

        super().__init__(adata)

        n_labels = self.summary_stats.n_labels
        n_batch = self.summary_stats.n_batch

        self.init_params_ = self._get_init_params(locals())

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=n_labels,
            n_latent_d=n_latent_d,
            n_latent_y=n_latent_y,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            **kwargs
        )

        self._module_kwargs = {
            "n_latent_d": n_latent_d,
            "n_latent_y": n_latent_y,
            "dropout_rate": dropout_rate,
            "dispersion": dispersion,
            "gene_likelihood": gene_likelihood,
            "latent_distribution": latent_distribution,
            **kwargs
        }

        self._model_summary_string = (
            "DiSCVI model with the following parameters: \n"
            f"n_latent_d: {n_latent_d}, n_latent_y: {n_latent_y}, "
            f"dropout_rate: {dropout_rate}, dispersion: {dispersion}, "
            f"gene_likelihood: {gene_likelihood}, latent_distribution: {latent_distribution}, "
        )

        if use_default_data_splitter:
            self._data_splitter_cls = DefaultDataSplitter

    # TODO: continue here!!!!

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        **kwargs,
    ):
        """
        <some doc string>
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False)
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

                Specifically, returns [ :math:`z_d` , :math:`z_y` ].

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
            q_zy_x = outputs["q_zy_x"]

            if give_mean:
                if self.module.latent_distribution == "ln":
                    samples_zd_x = q_zd_x.sample([mc_samples])
                    samples_zy_x = q_zy_x.sample([mc_samples])

                    zd_x = torch.nn.functional.softmax(samples_zd_x, dim=-1)
                    zd_x = zd_x.mean(dim=0)
                    zy_x = torch.nn.functional.softmax(samples_zy_x, dim=-1)
                    zy_x = zy_x.mean(dim=0)

                else:
                    zd_x = q_zd_x.loc
                    zy_x = q_zy_x.loc
            else:
                zd_x = outputs["zd_x"]
                zy_x = outputs["zy_x"]

            cat_vecs = [zd_x.cpu(), zy_x.cpu()]
            latent += [torch.cat(cat_vecs, dim=1)]

        return torch.cat(latent).numpy()

    def latent_separated(self, **kwargs):
        """Returns [ :math:`z_d` , :math:`z_y` ]
        Parameters
        ----------
        Returns
        -------
        """
        latent = self.get_latent_representation(**kwargs)
        n_d = self._module_kwargs["n_latent_d"]
        return latent[:, :n_d], latent[:, n_d:]

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
        if 'n_epochs_kl_warmup' not in plan_kwargs:
            plan_kwargs['n_epochs_kl_warmup'] = min(400, max_epochs)

        if 'n_epochs_warmup' not in plan_kwargs:
            plan_kwargs['n_epochs_ce_warmup'] = max_epochs

        training_plan = self._training_plan_cls(self.module, **plan_kwargs, n_classes=self.module.n_labels)

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
        self.module.init_kl_weights(self.adata, data_module.train_idx, self._label_key, self._batch_key)
        return runner()

    @classmethod
    @devices_dsp.dedent
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: Union[str, BaseModelClass],
        inplace_subset_query_vars: bool = False,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        unfrozen: bool = False,
        freeze_dropout: bool = False,
        freeze_expression: bool = True,
        freeze_decoder_first_layer: bool = True,
        freeze_batchnorm_encoder: bool = True,
        freeze_batchnorm_decoder: bool = False,
        freeze_classifier: bool = True,
    ):
        """TODO: some_doc_string    """
        _, _, device = parse_device_args(
            accelerator=accelerator,
            devices=device,
            return_device='torch',
            validate_single_device=True,
        )

        attr_dict, var_names, load_state_dict = _get_loaded_data(reference_model, device=device)

        _validate_var_names(adata, var_names)

        registry = attr_dict.pop("registry_")
        print(registry[_MODEL_NAME_KEY])
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] not in [cls.__name__, "SCDIVA"]:
            raise ValueError("It appears you are loading a model from a different class.")

        if _SETUP_ARGS_KEY not in registry:
            raise ValueError(
                "Saved model does not contain original setup inputs. "
                "Cannot load the original setup."
            )

        cls.setup_anndata(
            adata,
            source_registry=registry,
            extend_categories=True,
            allow_missing_labels=True,
            **registry[_SETUP_ARGS_KEY]
        )
        print(attr_dict)
        model = _initialize_model(cls, adata, attr_dict)
        adata_manager = model.get_anndata_manager(adata, required=True)
        version_split = adata_manager.registry[_SCVI_VERSION_KEY].split(".")
        if int(version_split[1]) < 8 and int(version_split[0]) == 0:
            warnings.warn(
                "Query integration should be performed using models trained with "
                "version >= 0.8",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )

        model.to_device(device)

        # model tweaking
        new_state_dict = model.module.state_dict()
        print(new_state_dict.keys())

        # we disable the classifiers
        for key, load_ten in load_state_dict.items():
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                continue
            # new categoricals changed size
            else:
                dim_diff = new_ten.size()[-1] - load_ten.size()[-1]
                fixed_ten = torch.cat([load_ten, new_ten[..., -dim_diff:]], dim=-1)
                load_state_dict[key] = fixed_ten

        model.module.load_state_dict(load_state_dict)
        model.module.eval()

        _set_params_online_update(
            model.module,
            unfrozen=unfrozen,
            freeze_decoder_first_layer=freeze_decoder_first_layer,
            freeze_batchnorm_encoder=freeze_batchnorm_encoder,
            freeze_batchnorm_decoder=freeze_batchnorm_decoder,
            freeze_dropout=freeze_dropout,
            freeze_expression=freeze_expression,
            freeze_classifier=freeze_classifier,
        )
        model.is_trained_ = False

        return model
