from typing import Optional, Literal

from anndata import AnnData

from scvi.dataloaders._data_splitting import DefaultDataSplitter
from scvi.model.base import ArchesMixin, RNASeqMixin, VAEMixin, BaseModelClass
from scvi.module._diva_rqm import RQMDiva


class ScDiVarQM(RNASeqMixin, VAEMixin, BaseModelClass, ArchesMixin):
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
