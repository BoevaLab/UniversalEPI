# UniversalEPI
UniversalEPI: Harnessing Attention Mechanisms to Decode Chromatin Interactions in Rare and Unexplored Cell Types

[![Preprint](https://img.shields.io/badge/preprint-available-green)](https://doi.org/10.1101/2024.11.22.624813) &nbsp;

<br/>

## Data Preprocessing

a. Input data processing
  - The details for processing ATAC-seq data from your raw input (BAM) or processed files (signal p-values bigwig and peaks bed) can be found in [`preprocessing/atac`](https://github.com/BoevaLab/UniversalEPI/tree/main/preprocessing/atac). This includes normalizing the bigwig and deduplication of ATAC-seq peaks.

b. Target data processing (optional)
  - [`preprocessing/hic`](https://github.com/BoevaLab/UniversalEPI/tree/main/preprocessing/hic) contains the details for processing Hi-C data from your raw input (.hic or .cool) or processed files (pairwise interaction files). This includes Hi-C normalization.
  - Combine ATAC-seq and Hi-C to create zigzag targets for each training cell line
    ```
    python ./preprocessing/prepare_target_data.py --cell_line gm12878 --atac_bed_path ./preprocessing/atac/data/GM12878_dedup.bed --hic_data_dir ./preprocessing/hic/data
    ```
  - Combine ATAC-seq and Hi-C to create zigzag targets for each test cell line
    ```
    python ./preprocessing/prepare_target_data.py --cell_line imr90 --atac_bed_path ./preprocessing/atac/data/IMR90_dedup.bed --hic_data_dir ./preprocessing/hic/data --test
    ```
    The above script will run for all autosomes (chr1-22) by default. The Hi-C resolution is assumed to be 5Kb. The Hg38 genome version is considered by default. These can be modified using appropriate flags.

<br/>

## Stage 1

a. Inference
  - TBD

b. Training (optional)
  - TBD

## Stage 2

a. Inference
  - TBD

b. Training (optional)
  - TBD
