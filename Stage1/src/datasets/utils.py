import os
from typing import Dict, List, Optional, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyBigWig as bw
import torch
from Bio import SeqIO
from omegaconf.dictconfig import DictConfig
from torchvision.datasets.utils import download_url, extract_archive
from tqdm import tqdm

from src import utils

log = utils.get_logger(__name__)

NARROWPEAK_HEADER = [
    # Name of the chromosome (or contig, scaffold, etc.).
    "chrom",
    # The starting position of the feature in the chromosome or scaffold. The first base in a chromosome
    # is numbered 0.
    "chromStart",
    # The ending position of the feature in the chromosome or scaffold. The chromEnd base is not included
    # in the display of the feature. For example, the first 100 bases of a chromosome are defined as
    # chromStart=0, chromEnd=100, and span the bases numbered 0-99.
    "chromEnd",
    # Name given to a region (preferably unique). Use "." if no name is assigned.
    "name",
    # Indicates how dark the peak will be displayed in the browser (0-1000). If all scores were "'0"' when
    # the data were submitted to the DCC, the DCC assigned scores 1-1000 based on signal value. Ideally
    # the average signalValue per base spread is between 100-1000.
    "score",
    # +/- to denote strand or orientation (whenever applicable). Use "." if no orientation is assigned.
    "strand",
    # Measurement of overall (usually, average) enrichment for the region.
    "signalValue",
    # Measurement of statistical significance (-log10). Use -1 if no pValue is assigned.
    "pValue",
    # Measurement of statistical significance using false discovery rate (-log10). Use -1 if no qValue is
    # assigned.
    "qValue",
    # Point-source called for this peak; 0-based offset from chromStart. Use -1 if no point-source called.
    "peak",
]


def download_and_extract(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> str:

    download_root = os.path.expanduser(download_root)

    if extract_root is None:
        extract_root = download_root

    if not filename:
        filename = os.path.basename(url)

    # Check if extracted file already exists (we do not have a md5 for this unfortunately)
    if filename.endswith("gz"):
        fpath = os.path.join(extract_root, filename[:-3])

        if os.path.isfile(fpath):

            log.info(f"Skipping {filename}, already extracted")

            return fpath

    # Otherwise download, this will check if archive already downloaded
    download_url(url, download_root, filename, md5)

    # Extract if downloaded file is archive
    if filename.endswith("gz"):
        archive = os.path.join(download_root, filename)
        log.info(f"Extracting {archive} to {extract_root}")
        return extract_archive(archive, extract_root, remove_finished)

    return os.path.join(extract_root, filename)


def download(sources, download_dir):

    for subfolder, config in sources.items():

        if "url" in config.keys():

            download_and_extract(
                config.url,
                f"{download_dir}/{subfolder}",
                md5=config.md5,
                remove_finished=True,
            )

        else:
            download(config, f"{download_dir}/{subfolder}")


def _sequence_to_tensor(sequence):

    # Split sequence into bases (unicode, 4 bytes)
    base = np.array(sequence, dtype=str)

    # Make view on data and only take first byte (unicode > ascii)
    base_ascii = base.view(np.uint8)[0::4]

    # Some bit magic, converts
    # N to 0, A to 1, C to 2, T to 3, G to 4
    base_class = np.right_shift(np.bitwise_and(base_ascii + 2, 15), 1)

    return torch.from_numpy(base_class)


def _verify_chrom_length(genome_info, genome):
    return all(
        [
            genome[chrom_info.chrom].shape[0] == chrom_info.length
            for _, chrom_info in genome_info.iterrows()
        ]
    )


def parse_genome(
    source: str, destination: str, chromosomes: List[str], force: bool = False
) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor]]:

    log.info(f"Read genome info from {source}")
    genome = SeqIO.to_dict(SeqIO.parse(source, "fasta"))

    # Get genome info
    genome_info = pd.DataFrame(chromosomes, columns=["chrom"])
    genome_info["length"] = genome_info.apply(
        lambda x: len(genome[x.chrom].seq), axis=1
    )

    # Load genome from file if exists
    if not force and os.path.isfile(destination):
        log.info(f"Preprocessed genome found at {destination}")

        genome = torch.load(destination)

        if _verify_chrom_length(genome_info, genome):
            return genome_info, genome

    # If we reach this, genome does not exist or not match chromosome length
    log.info("Preprocess genome ...")
    genome_dict = {
        chrom: _sequence_to_tensor(genome[chrom].seq.upper())
        for chrom in tqdm(chromosomes, unit="chr")
    }

    assert _verify_chrom_length(
        genome_info, genome_dict
    ), "Decompressed genome does not satisfy required chromosome length"

    log.info(f"Save preprocessed genome to {destination}")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    torch.save(genome_dict, destination)

    return genome_info, genome


def parse_dnase(
    source: str, destination: str, genome_info: pd.DataFrame, force: bool = False
) -> Dict[str, torch.Tensor]:

    log.info(f"Read dnase from {source}")

    # Load genome from file if exists
    if not force and os.path.isfile(destination):
        log.info(f"Preprocessed dnase found at {destination}")

        dnase_dict = torch.load(destination)

        if _verify_chrom_length(genome_info, dnase_dict):
            return dnase_dict

    # If we reach this, dnase file does not exist or not match chromosome length
    log.info("Preprocess dnase ...")

    genome_info["end"] = genome_info.length.cumsum()
    genome_info["start"] = genome_info.end - genome_info.length

    dnase = torch.zeros(genome_info.end.max(), dtype=torch.float32)

    with bw.open(source) as signal:
        for _, chrom in tqdm(
            genome_info.iterrows(), unit="chr", total=genome_info.shape[0]
        ):
            dnase[chrom.start : chrom.end] = torch.Tensor(
                signal.values(chrom.chr, 0, chrom.length)
            )

    log.info("Normalize dnase ...")

    # Log normalize
    dnase = dnase.nan_to_num()
    dnase = (dnase + 1).log()
    mean, std = torch.mean(dnase), torch.std(dnase)
    dnase = (dnase - mean) / std

    # Save as float 16
    dnase = dnase.type(torch.float16)
    dnase_dict = {
        chrom.chr: dnase[chrom.start : chrom.end] for _, chrom in genome_info.iterrows()
    }

    assert _verify_chrom_length(
        genome_info, dnase_dict
    ), "Decompressed dnase does not satisfy required chromosome length"

    log.info(f"Save preprocessed dnase to {destination}")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    torch.save(dnase_dict, destination)

    return dnase_dict


def parse_mappability(
    source: str, destination: str, genome_info: pd.DataFrame, force: bool = False
) -> Dict[str, torch.Tensor]:

    log.info(f"Read mappability from {source}")

    # Load genome from file if exists
    if not force and os.path.isfile(destination):
        log.info(f"Preprocessed mappability found at {destination}")

        map_dict = torch.load(destination)

        if _verify_chrom_length(genome_info, map_dict):
            return map_dict

    # If we reach this, mappability file does not exist or not match chromosome length
    log.info("Preprocess mappability ...")

    genome_info["end"] = genome_info.length.cumsum()
    genome_info["start"] = genome_info.end - genome_info.length

    mappability = torch.zeros(genome_info.end.max(), dtype=torch.float32)
    with bw.open(source) as signal:
        for _, chrom in tqdm(
            genome_info.iterrows(), unit="chr", total=genome_info.shape[0]
        ):
            log.info(f"{chrom.chrom}; {chrom.length}")
            mappability[chrom.start : chrom.end] = torch.Tensor(
                signal.values(chrom.chrom, 0, chrom.length)
            )

    # Save as float 16
    mappability = mappability.type(torch.float16)
    map_dict = {
        chrom.chrom: mappability[chrom.start : chrom.end] for _, chrom in genome_info.iterrows()
    }

    assert _verify_chrom_length(
        genome_info, map_dict
    ), "Decompressed mappability does not satisfy required chromosome length"

    log.info(f"Save preprocessed mappability to {destination}")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    torch.save(map_dict, destination)

    return map_dict


def parse_index(
    source_path: str,
    genome_info: pd.DataFrame,
    bin_width: int = 1000,
    bin_step: int = 200,
    thresholds: DictConfig = None,
) -> pd.DataFrame:

    log.info(f"Read peaks from {source_path}")

    # Read all peaks from files
    peaks = dd.read_csv(
        f"{source_path}/*/*/*.narrowPeak",
        sep="\t",
        names=NARROWPEAK_HEADER,
        dtype={"qValue": "float64", "signalValue": "float64"},
        include_path_column="full_path",
    )

    peaks[["root", "data_type", "cell_line", "file_name"]] = peaks.full_path.str.rsplit(
        "/", expand=True, n=3
    )

    # Filter down to selected chromosomes
    peaks = peaks[peaks.chrom.isin(genome_info.chrom)]

    # Apply thresholds for each data type
    if thresholds is not None:
        for data_type, threshold in thresholds.items():
            peaks = peaks[
                ~((peaks.data_type == data_type) & (peaks.signalValue <= threshold))
            ]

    # Calculate center bin
    peaks["center"] = peaks.chromStart + 0.5 * (peaks.chromEnd - peaks.chromStart)

    # Bin ids > 0 so we can truncate instead of floor
    peaks["center_bin"] = (peaks.center / bin_step).astype(int)

    # Calculate overlapping bins
    peaks["bin"] = peaks.apply(
        lambda x: list(
            range(x.center_bin - int(bin_width / bin_step) + 1, x.center_bin + 1)
        ),
        axis=1,
        meta=("bin", "object"),
    )

    # Filter down to required columns
    peaks = peaks[["cell_line", "chrom", "data_type", "bin"]]

    # Process to memory
    log.info("Preprocess peaks...")
    peaks = peaks.compute()

    log.info("-" * 20)
    log.info(
        f"""Number of peaks:\n\n{
        pd.pivot_table(peaks, values='bin', index='cell_line', columns='data_type', aggfunc=len)
        }\n"""
    )
    log.info("-" * 20)

    # Explode by bin
    peaks = peaks.explode("bin")

    # Pivot by data type
    peaks["peak"] = True
    peaks = peaks.reset_index(drop=True)
    peaks = peaks.pivot_table(
        index=["cell_line", "chrom", "bin"],
        columns="data_type",
        values="peak",
        aggfunc="any",
        fill_value=False,
    )

    peaks = peaks.reset_index(drop=False)

    log.info("Done")

    return peaks
