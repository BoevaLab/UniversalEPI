"""
npz_dataset.py

Dataset class to load npz files
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NPZDatasetRaw(Dataset):
    def __init__(self, data_dir=None, data=None, transform=None):
        """
        Args:
            data_dir: npz file path
            data: npz file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if data_dir:
            data = np.load(data_dir)
        elif data is None:
            raise ValueError("Data is not provided")

        self.transform = transform  # if apply any transformation
        self.FLANK = 200
        self.indexing = data["indexing"]
        self.dnase = data["dnase"]
        self.sequence = data["sequence"]
        self.meta = data["meta"]
        self.mappability = data["mappability"]

        if "blacklist" in data.keys():
            self.blacklist = data["blacklist"]
        else:
            self.blacklist = None
        if "target" in data.keys():
            self.target = data["target"]
        else:
            self.target = None

    def __len__(self):
        return len(self.indexing) - 2 * self.FLANK

    def __getitem__(self, idx):

        data_idx = self.indexing[idx]
        x_dnase = torch.from_numpy(np.array(self.dnase[data_idx - self.FLANK : data_idx + self.FLANK + 1, :])).float()
        x_map = torch.from_numpy(
            np.array(self.mappability[data_idx - self.FLANK : data_idx + self.FLANK + 1, :])
        ).float()

        x_seq = np.reshape(np.array(self.sequence[data_idx - self.FLANK : data_idx + self.FLANK + 1, :]), [-1, 4, 1000])
        x_seq = torch.from_numpy(x_seq).float()

        if self.target is not None:
            y = torch.from_numpy(np.squeeze(np.array(self.target[idx, :]))).float()
        else:
            y = torch.zeros(1)

        if self.blacklist is None:
            mask = torch.zeros(1)
        else:
            mask = torch.from_numpy(np.squeeze(np.array(self.blacklist[idx, :]))).float()
        m = torch.from_numpy(np.array(self.meta[data_idx - self.FLANK : data_idx + self.FLANK + 1, :])).float()

        if self.transform:
            x = self.transform(x)

        return x_dnase, x_seq, y, m, x_map, mask
