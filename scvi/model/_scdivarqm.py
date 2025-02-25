import logging
import warnings
from typing import Literal, Union, Callable, Optional, Sequence

import scvi.nn
import torch
from anndata import AnnData
from scvi import settings
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY, _SCVI_VERSION_KEY
from scvi.dataloaders import SemiSupervisedDataLoader
from scvi.dataloaders._data_splitting import MixedRatioDataSplitter
from scvi.model import SCDIVA
from scvi.model._utils import parse_device_args
from scvi.model.base import ArchesMixin, BaseModelClass
from scvi.model.base._archesmixin import _get_loaded_data
from scvi.model.base._utils import _validate_var_names, _initialize_model
from scvi.module._diva_rqm import RQMDiva
from scvi.utils._docstrings import devices_dsp
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ScDiVarQM(SCDIVA, ArchesMixin):
    """
    Streamlined RQM implementation of scDIVA
    """
    _module_cls = RQMDiva

    def __init__(
        self,
        adata: AnnData | None = None,
        n_latent_d: int = 4,
        n_latent_y: int = 10,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        use_ratio_data_splitter=False,
        use_default_data_splitter=False,
        **kwargs
    ):
        if use_ratio_data_splitter:
            self._data_splitter_cls = MixedRatioDataSplitter
        super_kwargs = {
            "alpha_d": 0,  # deactivate the batch classifier
            # "arches_batch_extension_size": number_query_batches  # TODO: implement
            # TODO: add any other parameters here if required
        }
        skwarg_keys = super_kwargs.keys()
        super().__init__(adata, n_latent_d, n_latent_y, dropout_rate, dispersion, gene_likelihood, latent_distribution,
                         use_default_data_splitter,
                         **self._filter_dict(kwargs, lambda x, _: x not in skwarg_keys), **super_kwargs)
        # todo: do extension of module with newly injected batch stuff for batch encoders to be able to perform scARCHES
        # some code here!

    @staticmethod
    def _filter_dict(filterable: dict, f: Callable[[any, any], bool]) -> dict:
        return {x: y for x, y in filterable.items() if f(x, y)}

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs, init_cls_weights=False)

    @classmethod
    @devices_dsp.dedent
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: Union[str, BaseModelClass],
        pred_type: Literal['prior_based', 'internal_classifier'] = 'internal_classifier',
        conservativeness: float = 0.5,
        inplace_subset_query_vars: bool = False,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        unfrozen: bool = False,
        freeze_dropout_prior_encoder: bool = False,
        freeze_expression: bool = True,
        freeze_decoder_first_layer: bool = True,
        freeze_batchnorm_encoder: bool = True,
        freeze_batchnorm_decoder: bool = False,
        unlabeled_category: str = "unknown",
        freeze_classifier: bool = True,
        use_ratio_data_splitter: bool = False
    ):
        """TODO: some_doc_string    """
        _, _, device = parse_device_args(
            accelerator=accelerator,
            devices=device,
            return_device='torch',
            validate_single_device=True,
        )

        attr_dict, var_names, load_state_dict = _get_loaded_data(reference_model, device=device)
        attr_dict["init_params_"]['kwargs']['kwargs'].update(dict(
            label_generator=pred_type,
            conservativeness=conservativeness,
        ))

        _validate_var_names(adata, var_names)

        registry = attr_dict.pop("registry_")
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] not in [cls.__name__, "SCDIVA"]:
            raise ValueError("It appears you are loading a model from an unsupported class.")

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
        attr_dict['init_params_']['kwargs']['kwargs'].update({'use_ratio_data_splitter': use_ratio_data_splitter})
        # attr_dict['init_params_']['kwargs']['kwargs'].update({'beta_y': 1})
        model: ScDiVarQM = _initialize_model(cls, adata, attr_dict)
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
        new_state_dict: dict = model.module.state_dict()
        allowed_keys = new_state_dict.keys()

        # we disable the classifiers
        load_target = {}
        for key, load_ten in load_state_dict.items():
            if not key in allowed_keys:
                continue
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                load_target[key] = load_ten
            else:
                dim_diff = new_ten.size()[-1] - load_ten.size()[-1]
                fixed_ten = torch.cat([load_ten, new_ten[..., -dim_diff:]], dim=-1)
                load_target[key] = fixed_ten

        model.module.load_state_dict(load_target)
        model.module.eval()

        _set_params_online_update(
            model.module,
            unfrozen=unfrozen,
            freeze_batchnorm_encoder=freeze_batchnorm_encoder,
            freeze_batchnorm_decoder=freeze_batchnorm_decoder,
            freeze_dropout_celltype_prior=freeze_dropout_prior_encoder,
            freeze_classifier=freeze_classifier
        )
        model.is_trained_ = False
        return model

    def get_reconstruction_likelihood(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_mc_samples: int = 1000,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> torch.tensor:
        """
        Used for sample selection.
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_mc_samples
            Number of Monte Carlo samples to use for marginal LL estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=SemiSupervisedDataLoader
        )
        if hasattr(self.module, "sample_reconstruction_ll"):
            p_x_zl_sum = []
            for tensors in tqdm(scdl):
                px_zd_zy = self.module.sample_reconstruction_ll(tensors, n_mc_samples=n_mc_samples, **kwargs)

                # bucket divide

                p_x_zl_sum.append(px_zd_zy)

            return torch.cat(p_x_zl_sum)
        else:
            raise NotImplementedError(
                "sample_reconstruction_ll is not implemented for current model. "
                "Please raise an issue on github if you need it."
            )


def _process_layer(layer: torch.nn.Sequential, freeze_grads: bool, freeze_batchnorm: bool, freeze_dropout: bool):
    layer_components: list[torch.nn.Module] = [*layer.children()]
    layer_components[0].requires_grad_(not freeze_grads)
    layer_components[1].requires_grad_(not freeze_batchnorm)
    layer_components[1].track_running_stats = not freeze_batchnorm
    if freeze_dropout:
        layer_components[-1].p = 0


def _set_module_freeze_state(mod: scvi.nn.Encoder | scvi.nn.DecoderSCVI,
                             freeeze_gradients: bool,
                             apply_scArches: bool,
                             freeze_batchnorm: bool,
                             freeze_dropout: bool):
    if not freeeze_gradients and apply_scArches:
        freeeze_gradients = True
        logger.warn('As scArches is applied, gradients will be frozen ignoring freeze_gradients=False')
    for child in mod.children():
        if isinstance(child, scvi.nn.FCLayers):
            layers = child.fc_layers
            if apply_scArches:  # only call function if necessary
                child.set_online_update_hooks()
                _process_layer(layers[0], False, freeze_batchnorm, freeze_dropout)
                layers = layers[1:]
            for layer in layers:
                _process_layer(layer, freeeze_gradients, freeze_batchnorm, freeze_dropout)
        else:
            child.requires_grad_(not freeeze_gradients)


def _set_params_online_update(
    module: RQMDiva,
    unfrozen: bool,
    freeze_batchnorm_encoder: bool,
    freeze_batchnorm_decoder: bool,
    freeze_dropout_celltype_prior: bool,
    freeze_classifier: bool
):
    """Freeze parts of network for scArches."""
    # do nothing if unfrozen
    if unfrozen:
        return

    module.px_r.requires_grad = False

    for name, module in module.named_children():
        """
        Three cases
        1. completely unfrozen
        2. completely frozen
        3. apply scARCHES to first layer
        """
        match name:
            # decoder: batch predecoder batchnorm may be unfrozen
            # same applies to joined decoder batchnorm
            case 'reconstruction_dxy_decoder':
                for name_, child in module.named_children():
                    print(name_)
                    is_batch_decoder = 'batch' in name_
                    print(is_batch_decoder)
                    k = 'joined' in name_
                    _set_module_freeze_state(child, not is_batch_decoder, False, freeze_batchnorm_decoder, False)
            case 'aux_y_zy_enc':
                module.requires_grad_(not freeze_classifier)
            case 'posterior_zy_x_encoder':
                # completely unfrozen, freeze batch-norm if requested
                _set_module_freeze_state(module, False, False, freeze_batchnorm_encoder, False)
            case 'posterior_zd_x_encoder':
                # completely unfrozen, freeze batch-norm if requested
                _set_module_freeze_state(module, False, False, freeze_batchnorm_encoder, False)
            case 'prior_zy_y_encoder':
                # frozen! (should not contain any batch-norm)
                _set_module_freeze_state(module, True, False, True, freeze_dropout_celltype_prior)
            case 'prior_zd_d_encoder':
                # apply scARCHES to first layer and freeze batch-norm if requested
                _set_module_freeze_state(module, True, True, freeze_batchnorm_encoder, False)
            case _:
                raise ValueError(f'Model contains layer: {name} which it should not! Make sure the model has not been corrupted')
