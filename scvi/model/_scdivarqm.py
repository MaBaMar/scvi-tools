import warnings
from typing import Literal, Union, Callable

import torch
from anndata import AnnData
from scvi import settings, REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY, _SCVI_VERSION_KEY
from scvi.data.fields import LayerField, CategoricalObsField, LabelsWithUnlabeledObsField
from scvi.data.fields import NumericalObsField
from scvi.model import SCDIVA
from scvi.model._utils import parse_device_args
from scvi.model.base import ArchesMixin, BaseModelClass
from scvi.model.base._archesmixin import _get_loaded_data
from scvi.model.base._utils import _validate_var_names, _initialize_model
from scvi.module._diva_rqm import RQMDiva
from scvi.nn._base_components import FCLayers
from scvi.utils._docstrings import devices_dsp


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
        use_default_data_splitter=False,
        **kwargs
    ):
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
        unlabeled_category: str = "unknown"
    ):
        """TODO: some_doc_string    """
        _, _, device = parse_device_args(
            accelerator=accelerator,
            devices=device,
            return_device='torch',
            validate_single_device=True,
        )

        attr_dict, var_names, load_state_dict = _get_loaded_data(reference_model, device=device)
        attr_dict["init_params_"]['kwargs']['kwargs']['label_generator'] = pred_type

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
                continue
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
            freeze_dropout=freeze_dropout,
        )
        model.is_trained_ = False
        return model


def _set_params_online_update(
    module,
    unfrozen,
    freeze_batchnorm_encoder,
    freeze_batchnorm_decoder,
    freeze_dropout,
):
    # print(60 * "=")
    # print()
    """Freeze parts of network for scArches."""
    # do nothing if unfrozen
    if unfrozen:
        return

    mod_no_grad = {"prior_zy_y_encoder", "aux_y_zy_enc", "celltype_predecoder", "joined_decoder"}
    mod_no_hooks_yes_grad = {"l_encoder", "batch_predecoder", "posterior_zy_x_encoder", "posterior_zd_x_encoder"}
    no_hooks = mod_no_grad.union(mod_no_hooks_yes_grad)

    # those modules don't get gradient hooks
    def no_hook_cond(key):
        return key.split(".")[0] in no_hooks or key.split(".")[1] in no_hooks

    def requires_grad(key):
        mod_name = key.split(".")[0]
        # set decoder layer that needs grad
        if "reconstruction" in mod_name:
            mod_name = key.split(".")[1]
        first_layer_of_grad_mod = "fc_layers" in key and ".0." in key and mod_name not in mod_no_grad
        # modules that need grad
        mod_force_grad = mod_name in mod_no_hooks_yes_grad
        is_non_frozen_batchnorm = (
                                      "fc_layers" in key
                                      and ".1." in key
                                      and "encoder" in key
                                      and (not freeze_batchnorm_encoder)
                                  ) or (
                                      "fc_layers" in key
                                      and ".1." in key
                                      and "decoder" in key
                                      and (not freeze_batchnorm_decoder)
                                  )

        return first_layer_of_grad_mod | mod_force_grad | is_non_frozen_batchnorm

    for key, mod in module.named_modules():
        # skip protected modules
        if key.split(".")[0] in mod_no_hooks_yes_grad:
            continue
        if isinstance(mod, FCLayers):
            mod.set_online_update_hooks(not no_hook_cond(key))
            # if not no_hook_cond(key):
            # print("Hooked:", key)
        if isinstance(mod, torch.nn.Dropout):
            if freeze_dropout:
                mod.p = 0
        # momentum freezes the running stats of batchnorm
        freeze_batchnorm = ("decoder" in key and freeze_batchnorm) or (
            "encoder" in key and freeze_batchnorm
        )
        if isinstance(mod, torch.nn.BatchNorm1d) and freeze_batchnorm:
            mod.momentum = 0

    for key, par in module.named_parameters():
        par.requires_grad = requires_grad(key)
