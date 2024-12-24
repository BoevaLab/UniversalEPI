from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from src import utils
from src.datasets import CellDataset

log = utils.get_logger(__name__)


class MultiCellModule(LightningDataModule):
    def __init__(
        self,
        mode: str,
        datasets: ListConfig,
        dataset_split: DictConfig,
        selected_targets: List[str],
        binning: DictConfig,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        force_reload: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self._dataset: Dict[str, ConcatDataset] = {}
        self._transforms: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._setup_complete: bool = False
        self._target_weight: torch.Tensor = None

        self.batch_size = batch_size

    def prepare_data(self) -> None:
        log.info("Prepare data")

    @property
    def binarize_targets(self):
        return self.hparams.mode == "classification"

    @property
    def dataset(self):
        return self._dataset

    def setup(self, stage: Optional[str] = None) -> None:

        if stage in (None, "fit", "validate"):
            # Load fit and validate stage together because the current
            # pytorch lightning version only calls with 'fit' for both stages
            self._load_stage("fit")
            self._load_stage("validate")

        if stage in (None, "test"):
            self._load_stage("test")

        if stage in (None, "predict"):
            self._load_stage("predict")

    def _load_stage(self, stage: str) -> None:
        log.info(f"Load stage {stage}")

        cell_lines = self.hparams.dataset_split.cell_line

        if stage not in self._dataset.keys():
            self._dataset[stage] = ConcatDataset(
                [
                    self._load_dataset(
                        cell_line,
                        stage,
                        cell_id=i if stage == "fit" else None
                    ) for i, cell_line in enumerate(cell_lines[stage])
                ]
            )

    def _load_dataset(self, cell_line, stage: str = "fit", cell_id=None) -> CellDataset:

        dataset: CellDataset = hydra.utils.instantiate(
            self.hparams.datasets[cell_line],
            cell_line=cell_line,
            bin_width=self.hparams.binning.width,
            bin_step=0 if stage == "predict" else self.hparams.binning.step,
            selected_targets=self.hparams.selected_targets,
            split=self.hparams.dataset_split.chromosome[stage],
            force_reload=self.hparams.force_reload,
            binarize_targets=self.hparams.mode == "classification",
            transforms=None if stage == "fit" else self._load_transforms(cell_line),
            _recursive_=False,
            cell_id=cell_id if stage == "fit" else -1,
        )

        dataset.setup()

        if stage == "fit":
            self._transforms[cell_line] = dataset.transforms

        return dataset
    
    def set_transforms(self, transforms):
        self._transforms = transforms

    def get_transforms(self):
        return self._transforms


    def _load_transforms(self, cell_line) -> Tuple[torch.Tensor, torch.Tensor]:

        if cell_line in self._transforms.keys():
            return self._transforms[cell_line]

        return (
            torch.mean(
                torch.stack([transform[0] for transform in self._transforms.values()])
            ),
            torch.mean(
                torch.stack([transform[1] for transform in self._transforms.values()])
            ),
        )

    @property
    def target_weight(self):

        if self._target_weight is None:
            self._load_stage("fit")

            num_positives = [
                dataset.targets[self.hparams.selected_targets].astype(bool).sum(axis=0)
                for dataset in self._dataset["fit"].datasets
            ]

            num_positives = pd.concat(num_positives).sum(axis=0)
            num_negatives = len(self._dataset["fit"]) - num_positives

            self._target_weight = torch.tensor(
                num_negatives / num_positives, dtype=torch.float32
            )

        return self._target_weight

    def train_dataloader(self):
        return DataLoader(
            dataset=self._dataset["fit"],
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self._dataset["validate"],
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self._dataset["test"],
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self._dataset["predict"],
            batch_size=self.batch_size,
            num_workers=0,  # self.hparams.num_workers,
            pin_memory=False,  # self.hparams.pin_memory,
            shuffle=False,
        )

