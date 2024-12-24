import os
import warnings
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyBigWig as bw
import torch
from Bio import SeqIO
from omegaconf.dictconfig import DictConfig
from pyjaspar import jaspardb
from torch.nn import Conv1d, Parameter
from torch.nn.functional import one_hot
from torchvision.datasets.utils import download_url, extract_archive
from tqdm import tqdm

from src import utils

log = utils.get_logger(__name__)

from functools import cached_property


class OnlineResource:
    def __init__(
        self, source: DictConfig, destination: DictConfig, force_reload: bool = False
    ) -> None:

        # Perform basic configuration checks
        self._check_source_config(source)
        self._check_destination_config(destination)

        # Save properties
        self._source = source
        self._destination = destination
        self._force_reload = force_reload

        # Download and extract resource
        self._download_and_extract()

    @property
    def remotename(self) -> str:
        return os.path.basename(self._source.url)

    @property
    def process_file_extension(self) -> str:
        return "pt"

    @property
    def url(self) -> str:
        return self._source.url

    @property
    def md5(self) -> str:
        return self._source.md5

    @property
    def download_root(self) -> str:
        return os.path.expanduser(self._destination.raw)

    @property
    def extract_root(self) -> str:
        return os.path.expanduser(self._destination.raw)

    @property
    def process_root(self) -> str:
        return os.path.expanduser(self._destination.processed)

    @property
    def download_path(self) -> str:
        return self.download_root

    @property
    def extract_path(self) -> str:
        return self.extract_root

    @property
    def process_path(self) -> str:
        return self.process_root

    @property
    def download_file(self) -> str:
        return os.path.join(self.download_path, self.remotename)

    @property
    def extract_file(self) -> str:
        return os.path.join(
            self.extract_path,
            self.remotename[:-3] if self.is_archive else self.remotename,
        )

    @property
    def process_file(self) -> str:
        filename = self.remotename[:-3] if self.is_archive else self.remotename

        if self.process_file_extension is not None:
            filename = f"{filename}.{self.process_file_extension}"

        return os.path.join(self.process_path, filename)

    @property
    def is_archive(self) -> bool:
        return self.remotename.endswith("gz")

    @property
    def force_reload(self) -> bool:
        return self._force_reload

    def _download_and_extract(self) -> None:

        # Download, this will check if archive already downloaded
        download_url(self.url, self.download_path, self.remotename, self.md5)

        # Extract if downloaded file is archive
        # and not already extracted
        if self.is_archive and (
            self.force_reload or not os.path.isfile(self.extract_file)
        ):
            log.info(f"Extracting {self.remotename} to {self.extract_path}")
            os.makedirs(self.extract_path, exist_ok=True)
            extract_archive(from_path=self.download_file, to_path=self.extract_path)

    @staticmethod
    def _check_source_config(source: DictConfig) -> None:
        assert (
            source is not None
        ), "Configuration error: Please provide a valide source configuration"

        assert (
            source.url is not None
        ), "Configuration error: No source url found in \{source.url\}"

        # assert (
        #     source.md5 is not None
        # ), "Configuration error: No source md5 found in \{source.md5\}"

    @staticmethod
    def _check_destination_config(destination: DictConfig) -> None:
        assert (
            destination is not None
        ), "Configuration error: Please provide a valide destination configuration"

        assert (
            destination.raw is not None
        ), """Configuration error: Please specify destination path for downloaded files 
                in \{destination.raw\}"""

        assert (
            destination.processed is not None
        ), """Configuration error: Please specify destination path for preprocessed data in
            in \{destination.processed\}"""


class SpeciesResource(OnlineResource):
    def __init__(
        self,
        species: str,
        chromosomes: List[str],
        source: DictConfig,
        destination: DictConfig,
        force_reload: bool = False,
    ) -> None:

        self._species = species
        self._chromosomes = list(chromosomes)

        super().__init__(source, destination, force_reload=force_reload)

    @property
    def chromosomes(self):
        return self._chromosomes

    @property
    def species(self) -> str:
        return self._species

    # Override OnlineResource paths
    @property
    def download_path(self) -> str:
        # Override the path where the genome is downloaded to
        return os.path.expanduser(f"{self.download_root}/{self.species}/genome")

    @property
    def extract_path(self) -> str:
        # Override the path where the genome is extracted to
        return os.path.expanduser(f"{self.extract_root}/{self.species}/genome")

    @property
    def process_path(self) -> str:
        # Override the path where the genome is processed to
        return os.path.expanduser(f"{self.process_root}/{self.species}/genome")


class SignalResource(SpeciesResource):
    def __init__(
        self,
        cell_line: str,
        data_type: str,
        species: str,
        chromosomes: List[str],
        source: DictConfig,
        destination: DictConfig,
        force_reload: bool = False,
    ) -> None:

        self._cell_line = cell_line
        self._data_type = data_type

        super().__init__(
            species, chromosomes, source, destination, force_reload=force_reload
        )

    @property
    def cell_line(self) -> str:
        return self._cell_line

    @property
    def data_type(self) -> str:
        return self._data_type

    # Override paths
    @property
    def download_path(self) -> str:
        # Override the path where the genome is downloaded to
        return os.path.expanduser(
            f"{self.download_root}/{self.species}/{self.cell_line}/{self.data_type}"
        )

    @property
    def extract_path(self) -> str:
        # Override the path where the genome is extracted to
        return os.path.expanduser(
            f"{self.extract_root}/{self.species}/{self.cell_line}/{self.data_type}"
        )

    @property
    def process_path(self) -> str:
        # Override the path where the genome is processed to
        return os.path.expanduser(
            f"{self.process_root}/{self.species}/{self.cell_line}/{self.data_type}"
        )


class FeatureSet:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self._stats: Dict[Dict[torch.Tensor]] = None

    def features(self, chromosomes: List[str] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def stats(self, chromosomes: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        log.info(f"Calculate statistics for chromosomes {chromosomes}")

        chromosome_data = self.features(chromosomes)
        signal = torch.cat([data for _, data in chromosome_data.items()], dim=0)

        mu, sigma = torch.std_mean(signal)

        log.info(f"Mu: {mu}, Sigma:{sigma}")

        return (mu, sigma)


class TargetSet:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self._stats: Dict[Dict[torch.Tensor]] = None

    COLUMNS = [
        "species",
        "cell_line",
        "chrom",
        "chromStart",
        "chromEnd",
        "center",
        "data_type",
        "strand",
        "signalValue",
        "original_index",
    ]

    class STRAND(Enum):
        BOTH = 0
        FORWARD = 1
        BACKWARD = -1

    def targets(self, chromosomes: List[str]) -> pd.DataFrame:
        raise NotImplementedError

    def stats(self, chromosomes: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        log.info(f"Calculate statistics for chromosomes {chromosomes}")

        values = self.targets(chromosomes).signalValue

        mu = values.mean()
        sigma = values.std()

        log.info(f"Mu: {mu}, Sigma:{sigma}")

        return (
            torch.tensor(mu, dtype=torch.float16),
            torch.tensor(sigma, dtype=torch.float16),
        )


class GenomeParser(FeatureSet, SpeciesResource):

    _genome: Dict[str, Dict[str, torch.Tensor]] = {}
    _genome_info: Dict[str, pd.DataFrame] = {}

    def __init__(
        self,
        species: str,
        chromosomes: List[str],
        source: DictConfig,
        destination: DictConfig,
        force_reload: bool = False,
    ) -> None:

        # First save list of chromosomes that need be parsed

        super().__init__(
            species, chromosomes, source, destination, force_reload=force_reload
        )

        if not species in GenomeParser._genome.keys():
            self.parse()

    @property
    def genome(self):
        warnings.warn("genome has been renamed to features")
        return self.features()

    @property
    def info(self):
        return GenomeParser._genome_info[self.species]

    # Implement FeatureSet interface
    def features(self, chromosomes: List[str] = None) -> Dict[str, torch.Tensor]:

        # If no filter specified, yield whole genome
        if chromosomes is None:
            return GenomeParser._genome[self.species]

        return {
            chromosome: GenomeParser._genome[self.species][chromosome]
            for chromosome in chromosomes
        }

    # Implement Parser interface
    def parse(self) -> None:

        INFO_EXT = ".info.pq"

        # Load genome from file if exists
        if (
            not self.force_reload
            and os.path.isfile(self.process_file)
            and os.path.isfile(self.process_file + INFO_EXT)
        ):

            log.info(f"Preprocessed genome found at {self.process_file}")

            genome_info = pd.read_parquet(self.process_file + INFO_EXT)
            genome_dict = torch.load(self.process_file)

            if self._verify_chrom_length(genome_info, genome_dict):
                GenomeParser._genome[self.species] = genome_dict
                GenomeParser._genome_info[self.species] = genome_info
                return

        # If we reach this, genome does not exist or not match chromosome length
        log.info("Preprocess genome ...")

        genome = SeqIO.to_dict(SeqIO.parse(self.extract_file, "fasta"))

        # Get genome info
        genome_info = pd.DataFrame(self.chromosomes, columns=["chrom"])
        genome_info["length"] = genome_info.apply(
            lambda x: len(genome[x.chrom].seq), axis=1
        )

        genome_dict = {
            chrom: self._sequence_to_tensor(genome[chrom].seq.upper())
            for chrom in tqdm(self.chromosomes, unit="chr")
        }

        assert self._verify_chrom_length(
            genome_info, genome_dict
        ), "Decompressed genome does not satisfy required chromosome length"

        log.info(f"Save preprocessed genome to {self.process_file}")
        os.makedirs(self.process_path, exist_ok=True)
        genome_info.to_parquet(self.process_file + INFO_EXT)
        torch.save(genome_dict, self.process_file)

        GenomeParser._genome[self.species] = genome_dict
        GenomeParser._genome_info[self.species] = genome_info

        return

    @staticmethod
    def _sequence_to_tensor(sequence):

        # Split sequence into bases (unicode, 4 bytes)
        base = np.array(sequence, dtype=str)

        # Make view on data and only take first byte (unicode > ascii)
        base_ascii = base.view(np.uint8)[0::4]

        # Some bit magic, converts
        # N to 0, A to 1, C to 2, T to 3, G to 4
        base_class = np.right_shift(np.bitwise_and(base_ascii + 2, 15), 1)

        return torch.from_numpy(base_class)

    @staticmethod
    def _verify_chrom_length(genome_info, genome):
        return all(
            [
                genome[chrom_info.chrom].shape[0] == chrom_info.length
                for _, chrom_info in genome_info.iterrows()
            ]
        )


class DnaseParser(FeatureSet, SignalResource):
    def __init__(
        self,
        genome: GenomeParser,
        cell_line: str,
        data_type: str,
        source: DictConfig,
        destination: DictConfig,
        force_reload: bool = False,
    ) -> None:

        self._genome_info: pd.DataFrame = genome.info
        self._signal: Dict[str, torch.Tensor] = None

        super().__init__(
            cell_line,
            data_type,
            genome.species,
            genome.chromosomes,
            source,
            destination,
            force_reload=force_reload,
        )

        self.parse()

    @property
    def info(self):
        return self._genome_info

    # Implement FeatureSet interface
    def features(self, chromosomes: List[str] = None) -> Dict[str, torch.Tensor]:

        # If no filter specified, yield whole signal
        if chromosomes is None:
            return self._signal

        return {chromosome: self._signal[chromosome] for chromosome in chromosomes}

    # Implement Parser interface
    def parse(self) -> None:

        log.info(f"Read signal from {self.extract_file}")

        # Load genome from file if exists
        if not self.force_reload and os.path.isfile(self.process_file):
            log.info(f"Preprocessed dnase found at {self.process_file}")

            dnase_dict = torch.load(self.process_file)

            if GenomeParser._verify_chrom_length(self.info, dnase_dict):
                self._signal = dnase_dict
                return

        # If we reach this, dnase file does not exist / not match chromosome length or we need to reload
        log.info("Preprocess signal ...")

        chomosome_info = self.info.copy()

        chomosome_info["end"] = chomosome_info.length.cumsum()
        chomosome_info["start"] = chomosome_info.end - chomosome_info.length

        dnase = torch.zeros(chomosome_info.end.max(), dtype=torch.float32)

        with bw.open(self.extract_file) as signal:
            for _, chrom in tqdm(
                chomosome_info.iterrows(), unit="chr", total=chomosome_info.shape[0]
            ):
                dnase[chrom.start : chrom.end] = torch.Tensor(
                    signal.values(chrom.chrom, 0, chrom.length)
                )

        # Log signal
        dnase = dnase.nan_to_num()
        dnase = (dnase + 1).log()

        # Save as float 16
        dnase = dnase.type(torch.float16)
        dnase_dict = {
            chrom.chrom: dnase[chrom.start : chrom.end]
            for _, chrom in chomosome_info.iterrows()
        }

        assert GenomeParser._verify_chrom_length(
            chomosome_info, dnase_dict
        ), "Signal does not satisfy required chromosome length"

        log.info(f"Save preprocessed signal to {self.process_file}")

        os.makedirs(self.process_path, exist_ok=True)
        torch.save(dnase_dict, self.process_file)

        self._signal = dnase_dict

        return


class NarrowPeakParser(TargetSet, SignalResource):

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

    def __init__(
        self,
        genome: GenomeParser,
        cell_line: str,
        data_type: str,
        source: DictConfig,
        destination: DictConfig,
        threshold: int = 0,
        force_reload: bool = False,
    ) -> None:

        self._threshold: int = threshold
        self._peaks: pd.DataFrame = None

        super().__init__(
            cell_line,
            data_type,
            genome.species,
            genome.chromosomes,
            source,
            destination,
            force_reload=force_reload,
        )

        self.parse()

    @property
    def threshold(self):
        return self._threshold

    # Overwrite file extension used for storage to reflect parquet export
    @property
    def process_file_extension(self):
        return "pq"

    # Implement TargetSet interface
    def targets(self, chromosomes: List[str] = None) -> pd.DataFrame:

        # If no filter specified, yield all peaks
        if chromosomes is None:
            return self._peaks

        return self._peaks[self._peaks.chrom.isin(chromosomes)]

    # Implement Parser interface
    def parse(self) -> None:

        log.info(f"Read narrow peak signal from {self.extract_file}")

        # Load parsed narrow peak from file if exists
        if not self.force_reload and os.path.isfile(self.process_file):
            log.info(f"Preprocessed narrow peak data found at {self.process_file}")
            peaks = pd.read_parquet(self.process_file)

            self._peaks = peaks
            return

        # Otherwise, read unprocessed narrow peak file from disk
        peaks: pd.DataFrame = pd.read_csv(
            self.extract_file,
            sep="\t",
            names=NarrowPeakParser.NARROWPEAK_HEADER,
            dtype={"qValue": "float64", "signalValue": "float64"},
        )

        # Filter down to selected chromosomes and apply threshold
        peaks = peaks[
            (peaks.chrom.isin(self.chromosomes)) & (peaks.signalValue > self.threshold)
        ]

        # Correct for narrow Peak format that starts counting at 1 instead of 0
        peaks.chromStart = peaks.chromStart - 1
        peaks.chromEnd = peaks.chromEnd - 1

        # Log transform
        peaks["signalValue"] = np.log(peaks.signalValue + 1)

        # Enrich labels with additional information about the signal
        peaks["species"] = self.species
        peaks["cell_line"] = self.cell_line
        peaks["data_type"] = self.data_type
        peaks["strand"] = self.STRAND.BOTH.value
        peaks["original_index"] = peaks.index

        # Calculate peak center
        peaks["center"] = (peaks.chromStart + peaks.peak).where(
            peaks.peak >= 0, peaks.chromStart + (peaks.chromEnd - peaks.chromStart) // 2
        )

        # Filter down to required columns
        peaks = peaks[self.COLUMNS]

        os.makedirs(self.process_path, exist_ok=True)
        peaks.to_parquet(self.process_file)

        self._peaks = peaks
        return


class CTCFParser(NarrowPeakParser):

    BACKGROUND = {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
    PSEUDOCOUNTS = {"A": 0.6, "C": 0.4, "G": 0.4, "T": 0.6}
    MOTIF_ID = "MA0139.1"
    THRESHOLD_FNR = 0.01
    DATATYPE_FORWARD = "ctcfpeakforward"
    DATATYPE_REVERSE = "ctcfpeakreverse"

    def __init__(
        self,
        genome: GenomeParser,
        cell_line: str,
        data_type: str,
        source: DictConfig,
        destination: DictConfig,
        threshold: int = 0,
        force_reload: bool = False,
    ) -> None:

        super().__init__(
            genome, cell_line, data_type, source, destination, threshold, force_reload
        )

        # Retrieve genome from gemome parser
        self._genome = genome.features()

        # Init convolutional layer
        self.conv = Conv1d(
            in_channels=self.kernel.shape[1],
            out_channels=self.kernel.shape[0],
            kernel_size=self.kernel.shape[2],
            padding="valid",
            bias=False,
        )

        self.conv.weight = Parameter(self.kernel, requires_grad=False)
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.conv.to(self.device)

        # Run strand detection on CTCF data
        self._strand_detection()

    @cached_property
    def motif(self):
        jdb_obj = jaspardb()
        motif = jdb_obj.fetch_motif_by_id(self.MOTIF_ID)
        motif.pseudocounts = self.PSEUDOCOUNTS
        motif.background = self.BACKGROUND

        return motif

    @cached_property
    def motif_score_threshold(self):
        distribution = self.motif.pssm.distribution(
            background=self.BACKGROUND, precision=10**4
        )
        return distribution.threshold_fnr(self.THRESHOLD_FNR)

    @cached_property
    def kernel(self):

        kernel_forward = torch.stack(
            [torch.tensor(self.motif.pssm[base]) for base in ["A", "C", "T", "G"]]
        )
        kernel_reverse = torch.stack(
            [
                torch.tensor(self.motif.reverse_complement().pssm[base])
                for base in ["A", "C", "T", "G"]
            ]
        )

        return torch.stack([kernel_forward, kernel_reverse])

    def _motif_score(self, x: pd.Series):

        # one-hot encode peak sequence and add batch dimension
        if torch.cuda.is_available():
            sequence_onehot = (
                one_hot(x.seq.long(), num_classes=5).T[1:5].float().unsqueeze(0).cuda()
            )
        else:
            sequence_onehot = (
                one_hot(x.seq.long(), num_classes=5).T[1:5].float().unsqueeze(0)
            )

        # calculate maximum motif score by applying convolution
        return torch.max(self.conv(sequence_onehot).squeeze(), dim=1).values.tolist()

    def _strand_detection(self):
        ctcf = self._peaks

        # Get sequence for each peak
        ctcf["seq"] = ctcf.apply(
            lambda x: self._genome[x.chrom][x.chromStart : x.chromEnd],
            axis=1,
            result_type="reduce",
        )

        with torch.no_grad():
            ctcf["motif_score"] = ctcf.apply(
                self._motif_score, axis=1, result_type="reduce"
            )

        ctcf["strand"] = [
            [self.STRAND.FORWARD.value, self.STRAND.BACKWARD.value] for _ in ctcf.index
        ]
        ctcf = ctcf.explode(["motif_score", "strand"])

        log.info(
            f"Motif score cutoff for FNR={self.THRESHOLD_FNR} is at {self.motif_score_threshold}"
        )

        ctcf = ctcf[ctcf.motif_score > self.motif_score_threshold]
        ctcf["data_type"] = np.where(
            ctcf.strand > 0, self.DATATYPE_FORWARD, self.DATATYPE_REVERSE
        )

        # Filter down to required columns
        self._peaks = ctcf[self.COLUMNS]
