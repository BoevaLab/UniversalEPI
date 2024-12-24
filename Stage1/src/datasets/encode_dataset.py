import glob
import hashlib
import warnings
from functools import cached_property
from typing import Dict, List, Tuple

import dask.dataframe as dd
import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from src import utils
from src.datasets import parse_mappability
from src.datasets.parser import GenomeParser, TargetSet

log = utils.get_logger(__name__)


# Note the half open intervals. We do not use X/Y chromosome
CHR = [f"chr{i}" for i in range(1, 23)]


class CellDataset(Dataset):
    def __init__(
        self,
        genome: DictConfig,
        cell_line: str,
        cell_id: int,
        data_types: DictConfig,
        selected_targets: List[str],
        split: DictConfig,
        bin_target: str,
        bin_width: int,
        bin_step: int = 0,
        force_reload: bool = False,
        binarize_targets: bool = False,
        transforms: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> None:

        super().__init__()

        # Each dataset is for one specific cell line
        self._cell_line = cell_line

        # ... and has exactly one associated genome configuration. This genome will be instantiated on first access
        self._genome_config = genome

        self.cell_id = cell_id

        # Data types, such as CTCF, make up the targets of this dataset. Like the genome, the will be instantiated on first access
        self._data_types_config = data_types

        # The features and targets yield only results for the chromosomes configured in the dataset split
        self._dataset_split_config = split

        # Some datatype might return more than one target (e.g. the dt CTCF has two). selected_targets holds the subset of
        # targets that will be loaded into memory
        self._selected_targets = list(selected_targets)

        # Binning properties
        self._bin_width = bin_width
        self._bin_step = (
            bin_step  # If 0, local instead of global binning will be applied,
        )
        self._bin_target = bin_target  # where the peaks called for bin_target defines the center of each bin

        # Modifiers for generated output
        self._transforms = transforms
        self._binarize_targets = binarize_targets

        # If true, re-downloads all data from source and forces all parsers through full pre-processing step
        self._force_reload = force_reload

    @property
    def dataset_split(self) -> List[str]:
        if self._dataset_split_config is None:
            return self._genome_config.chromosomes

        return list(self._dataset_split_config)

    @cached_property
    def transforms(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._transforms is None:
            return torch.std_mean(self.dnase)

        return self._transforms

    @property
    def use_global_binning(self) -> bool:
        return self._bin_step != 0

    @property
    def selected_targets(self) -> List[str]:
        return self._selected_targets

    @property
    def available_targets(self) -> List[str]:
        return [
            target_name
            for target_name in self.selected_targets
            if target_name in self.targets.columns
        ]

    @cached_property
    def _genome_parser(self) -> GenomeParser:
        log.info(f"Load genome for {self._genome_config.species}")

        return hydra.utils.instantiate(
            self._genome_config, force_reload=self._force_reload
        )

    @cached_property
    def genome(self) -> Dict[str, torch.Tensor]:
        return self._genome_parser.features(self.dataset_split)

    @cached_property
    def _all_parsers(self) -> Dict[str, TargetSet]:
        log.info(f"Load all features and targets for {self._cell_line}")

        genome_parser = self._genome_parser

        parsers = {
            data_type: hydra.utils.instantiate(
                parser_config,
                genome=genome_parser,
                cell_line=self._cell_line,
                data_type=data_type,
                force_reload=self._force_reload,
                _recursive_=False,
            )
            for data_type, parser_config in self._data_types_config.items()
        }

        return parsers

    @cached_property
    def targets(self) -> pd.DataFrame:

        target_parsers = self._all_parsers

        targets = pd.concat(
            [
                parser.targets(self.dataset_split)
                for _, parser in target_parsers.items()
                if isinstance(parser, TargetSet)
            ]
        )

        log.info(f"Apply {'global' if self.use_global_binning else 'local'} binning")
        targets = (
            self._global_binning(targets)
            if self.use_global_binning
            else self._local_binning(targets)
        )

        # Make sure bins do not overlap chromosome boundaries
        targets = targets.merge(self._genome_parser.info, on="chrom")
        targets = targets[(targets.bin_start >= 0) & (targets.bin_end < targets.length)]
        targets = targets.drop(columns="length")

        return targets

    @cached_property
    def dnase(self):

        log.info(f"Slice dnase signal")

        # Select subset of chromosomes for each feature that corresponds to the configured split
        # TODO: move source signal to config, can be atac or dnase
        try:
            dnasesignal = self._all_parsers["atacsignal"].features(self.dataset_split)
        except:
            dnasesignal = self._all_parsers["dnasesignal"].features(self.dataset_split)
        sliced_dnasesignal = self.targets.apply(
            lambda x: dnasesignal[x.chrom][x.bin_start : x.bin_end],
            axis=1,
            result_type="reduce",
        )
        sliced_dnasesignal = torch.stack(sliced_dnasesignal.to_list())

        return sliced_dnasesignal

    def setup(self):

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        
        include_features = self.available_targets + [self._bin_target]

        # Init targets
        all_peaks = self.targets[include_features].copy()

        # Number of samples per target
        positives: pd.Series = all_peaks.astype(bool).sum(axis=0)
        positives["Total"] = self.targets.shape[0]
        positives = positives.to_frame(name="samples")
        positives["coverage (%)"] = positives.samples * 100 // self.targets.shape[0]

        log.info("-" * 20)

        log.info(f"Number of samples:\n\n{positives.to_string()}\n")

        # Overall statistics like mean / avg
        log.info(f"Statistics:\n\n{all_peaks.mask(all_peaks == 0).describe()}\n")

        # Peaks overlapping binning target
        overlapping = all_peaks[all_peaks[self._bin_target] > 0][include_features]
        log.info(
            f"Peaks overlapping {self._bin_target}:\n\n{overlapping.mask(overlapping == 0).describe()}\n"
        )

        # Peaks not overlapping binning target
        non_overlapping = all_peaks[all_peaks[self._bin_target] == 0][include_features]
        log.info(
            f"Peaks not overlapping {self._bin_target}:\n\n{non_overlapping.mask(non_overlapping == 0).describe()}\n"
        )

        # Init dnase
        sigma, mu = torch.std_mean(self.dnase)

        log.info(f"Dnase has mean {mu} and std {sigma}")

        if self._transforms is not None:
            sigma, mu = self._transforms
            log.info(f"Using override mean {mu} and std {sigma}")
        
        log.info("Loading mappability")
        base_dir = f"{hydra.utils.get_original_cwd()}/data/"
        self.mappability = parse_mappability(
            f"{base_dir}/mappability/raw/k36.Umap.MultiTrackMappability.hg38.bw",
            f"{base_dir}/mappability/processed/k36.Umap.MultiTrackMappability.hg38.pt",
            # f"{base_dir}/mappability/raw/wgEncodeCrgMapabilityAlign36mer.bigWig",
            # f"{base_dir}/mappability/processed/wgEncodeCrgMapabilityAlign36mer.pt",
            self._genome_parser.info,
            self._force_reload,
        )


        log.info("-" * 20)

    def _global_binning(self, targets: pd.DataFrame) -> pd.DataFrame:

        targets["bin"] = targets.apply(
            lambda x: list(
                range(
                    (x.center - self._bin_width) // self._bin_step + 1,
                    x.center // self._bin_step + 1,
                )
            ),
            axis=1,
        )

        targets = targets.explode("bin")

        # drop chrom_start, chrom_end and center
        targets = targets.pivot_table(
            index=["species", "cell_line", "chrom", "bin"],
            columns="data_type",
            values="signalValue",
            aggfunc="max",
            fill_value=0,
        )

        targets = targets.reset_index(drop=False)

        targets["bin_start"] = targets.bin * self._bin_step
        targets["bin_end"] = targets.bin_start + self._bin_width

        return targets

    def _local_binning(self, targets: pd.DataFrame) -> pd.DataFrame:

        targets["bin_start"] = targets.center - self._bin_width // 2
        targets["bin_end"] = targets.bin_start + self._bin_width

        all_dnase_peaks = []
        all_other_peaks = []

        for _, data in targets.groupby(["species", "cell_line", "chrom"]):

            dnase_filter = data.data_type == self._bin_target

            dnase_peaks: pd.DataFrame = data[dnase_filter]
            dnase_peaks = (
                dnase_peaks.reset_index(drop=True)
                .reset_index(drop=False)
                .rename(columns={"index": "bin"})
            )

            other_peaks: pd.DataFrame = data[~dnase_filter]

            other_peaks["bin"] = other_peaks.apply(
                lambda x: dnase_peaks[
                    (dnase_peaks.center >= x.bin_start)
                    & (dnase_peaks.center <= x.bin_end)
                ].bin.values,
                axis=1,
                result_type="reduce",
            )

            other_peaks = other_peaks.explode("bin").dropna()

            all_dnase_peaks.append(dnase_peaks)
            all_other_peaks.append(other_peaks)

        all_dnase_peaks = pd.concat(all_dnase_peaks)
        all_other_peaks = pd.concat(all_other_peaks)
        all_peaks = pd.concat([all_dnase_peaks, all_other_peaks])

        all_peaks = all_peaks.pivot_table(
            index=["species", "cell_line", "chrom", "bin"],
            columns="data_type",
            values="signalValue",
            aggfunc="max",
            fill_value=0,
        )

        all_peaks = all_peaks.reset_index(drop=False)
        all_peaks = all_peaks.merge(
            all_dnase_peaks[
                [
                    "chrom",
                    "chromStart",
                    "chromEnd",
                    "center",
                    "bin",
                    "bin_start",
                    "bin_end",
                    "original_index",
                ]
            ],
            on=["chrom", "bin"],
        )

        return all_peaks

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, index):

        item = self.targets.iloc[index]

        item_sequence = self.genome[item.chrom][item.bin_start : item.bin_end]
        item_sequence_onehot = (
            one_hot(item_sequence.long(), num_classes=5).T[1:5].bool()
        )

        item_dnase = self.dnase[index]
        item_mappability = self.mappability[item.chrom][item.bin_start : item.bin_end]

        targets = item[self.available_targets]
        targets = targets.astype(bool) if self._binarize_targets else targets

        sigma, mu = self.transforms

        sample = {
            "dnase": (item_dnase - mu) / sigma,
            "sequence": item_sequence_onehot,
            "mappability": item_mappability,
            "target": torch.tensor(targets),
            "cell_id": self.cell_id,
        }

        return sample