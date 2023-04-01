"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random
import scipy.io
import h5py
from torch.utils.data import Dataset
from data import transforms as T
import torch

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            #kspace = h5py.File(fname, 'r')['kspace']
            original_image=h5py.File(fname, 'r')['reconstruction_esc']
            num_slices = original_image.shape[0]
            self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        mat_4 = scipy.io.loadmat('/home/itu-1001/Desktop/mask/mask_4.mat')
        mask_4 = mat_4['mask']
        mask_4 = T.to_tensor(mask_4).unsqueeze(-1)
        mask_4= torch.cat([mask_4,mask_4],-1)
        mat_8 = scipy.io.loadmat('/home/itu-1001/Desktop/mask/mask_8.mat')
        mask_8 = mat_8['mask_8']
        mask_8 = T.to_tensor(mask_8).unsqueeze(-1)
        mask_8= torch.cat([mask_8,mask_8],-1)
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace,mask_4, mask_8, target, data.attrs, fname.name, slice)
