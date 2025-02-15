import itertools
from typing import Optional, Union

import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data._utils import get_anndata_attribute

from ._ann_dataloader import AnnDataLoader
from ._concat_dataloader import ConcatDataLoader


class SemiSupervisedDataLoader(ConcatDataLoader):
    """DataLoader that supports semisupervised training.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    n_samples_per_label
        Number of subsamples for each label class to sample per epoch. By default, there
        is no label subsampling.
    indices
        The indices of the observations in the adata to load
    shuffle
        Whether the data should be shuffled
    batch_size
        minibatch size to load each iteration
    data_and_attributes
        Dictionary with keys representing keys in data registry (`adata_manager.data_registry`)
        and value equal to desired numpy loading type (later made into torch tensor).
        If `None`, defaults to all registered data.
    data_loader_kwargs
        Keyword arguments for :class:`~torch.utils.data.DataLoader`
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        n_samples_per_label: Optional[int] = None,
        indices: Optional[list[int]] = None,
        shuffle: bool = False,
        batch_size: int = 128,
        data_and_attributes: Optional[dict] = None,
        drop_last: Union[bool, int] = False,
        **data_loader_kwargs,
    ):
        adata = adata_manager.adata
        if indices is None:
            indices = np.arange(adata.n_obs)

        self.indices = np.asarray(indices)

        if len(self.indices) == 0:
            return None

        self.n_samples_per_label = n_samples_per_label

        labels_state_registry = adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        labels = get_anndata_attribute(
            adata_manager.adata,
            adata_manager.data_registry.labels.attr_name,
            labels_state_registry.original_key,
        ).ravel()

        # save a nested list of the indices per labeled category
        self.labeled_locs = []
        for label in np.unique(labels):
            if label != labels_state_registry.unlabeled_category:
                label_loc_idx = np.where(labels[indices] == label)[0]
                label_loc = self.indices[label_loc_idx]
                self.labeled_locs.append(label_loc)
        labelled_idx = self.subsample_labels()

        super().__init__(
            adata_manager=adata_manager,
            indices_list=[self.indices, labelled_idx],
            shuffle=shuffle,
            batch_size=batch_size,
            data_and_attributes=data_and_attributes,
            drop_last=drop_last,
            **data_loader_kwargs,
        )

    def resample_labels(self):
        """Resamples the labeled data."""
        labelled_idx = self.subsample_labels()
        # self.dataloaders[0] iterates over full_indices
        # self.dataloaders[1] iterates over the labelled_indices
        # change the indices of the labelled set
        self.dataloaders[1] = AnnDataLoader(
            self.adata_manager,
            indices=labelled_idx,
            shuffle=self._shuffle,
            batch_size=self._batch_size,
            data_and_attributes=self.data_and_attributes,
            drop_last=self._drop_last,
        )

    def subsample_labels(self):
        """Subsamples each label class by taking up to n_samples_per_label samples per class."""
        if self.n_samples_per_label is None:
            return np.concatenate(self.labeled_locs)

        sample_idx = []
        for loc in self.labeled_locs:
            if len(loc) < self.n_samples_per_label:
                sample_idx.append(loc)
            else:
                label_subset = np.random.choice(loc, self.n_samples_per_label, replace=False)
                sample_idx.append(label_subset)
        sample_idx = np.concatenate(sample_idx)
        return sample_idx


class SemiSupervisedFixedRatioDataLoader:
    """
    This dataloader is used for controlled ratio sampling for unsupervised training.

    Parameters
    ----------
    adata_manager
        The adata_manager object that has been created via ``setup_anndata``.
    indices
        The indices of the observations in the adata to load.
    shuffle
        Whether the data should be shuffled
    batch_size
        The minibatch size to load each iteration (includes both labeled and unlabeled data)
    supervised_ratio
        The ratio of supervised samples per minibatch
    data_and_attributes
        Dictionary with keys representing keys in data registry (`adata_manager.data_registry`)
        and value equal to desired numpy loading type (later made into torch tensor).
        If `None`, defaults to all registered data.
    data_loader_kwargs
        Keyword arguments for :class:`~torch.utils.data.DataLoader`
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        indices: Optional[list[int]] = None,
        shuffle: bool = False,
        batch_size: int = 128,
        supervised_ratio: float = 0.1,
        data_and_attributes: Optional[dict] = None,
        drop_last: Union[bool, int] = False, # TODO, find a way to use
        **data_loader_kwargs,
    ):
        self._curr_iter = 0
        if indices is None:
            indices = np.arange(adata_manager.adata.n_obs)

        self.indices = np.asarray(indices)

        if len(self.indices) == 0:
            return

        labels_state_registry = adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        labels = get_anndata_attribute(
            adata_manager.adata,
            adata_manager.data_registry.labels.attr_name,
            labels_state_registry.original_key,
        ).ravel()[indices]

        unlabeled_idx = labels == labels_state_registry.unlabeled_category
        labeled_idx = ~unlabeled_idx

        unlabeled_idx = indices[unlabeled_idx]
        labeled_idx = indices[labeled_idx]

        labeled_batch_sz = int(supervised_ratio * batch_size)
        unlabeled_batch_sz = batch_size - labeled_batch_sz

        self.unsup_loader = AnnDataLoader(
                adata_manager,
                indices=unlabeled_idx,
                shuffle=shuffle,
                batch_size=unlabeled_batch_sz,
                data_and_attributes=data_and_attributes,
                drop_last=True, # TODO: maybe change
                **data_loader_kwargs,
            )

        self.sup_loader = AnnDataLoader(
                adata_manager,
                indices=labeled_idx,
                shuffle=shuffle,
                batch_size=labeled_batch_sz,
                data_and_attributes=data_and_attributes,
                drop_last=True, # TODO: maybe change
                **data_loader_kwargs,
            )

        self._iter_max = max(len(self.unsup_loader), len(self.sup_loader))
        self._reset()

    def __iter__(self):
        self._reset()
        return self

    def _reset(self):
        self._curr_iter = 0
        self.unsup_loader_cycle = itertools.cycle(self.unsup_loader)
        self.sup_loader_cycle = itertools.cycle(self.sup_loader)

    def __next__(self):
        if self._curr_iter < self._iter_max:
            labeled_batch = next(self.unsup_loader_cycle)
            unlabeled_batch = next(self.sup_loader_cycle)
            self._curr_iter += 1
            return _combine_batches(labeled_batch, unlabeled_batch)
        raise StopIteration

    def __len__(self):
        return self._iter_max

def _combine_batches( batch_l: dict, batch_r: dict):
    if batch_l.keys() != batch_r.keys():
        raise ValueError("keys of dicts to merge do not match")
    for key in batch_l.keys():
        batch_l[key] = torch.concat([batch_l[key], batch_r[key]], dim=0)
    return batch_l

