# UniversalEPI
**UniversalEPI: Harnessing Attention Mechanisms to Decode Chromatin Interactions in Rare and Unexplored Cell Types**

[![Preprint](https://img.shields.io/badge/preprint-available-green)](https://doi.org/10.1101/2024.11.22.624813) &nbsp;
[![DOI](https://zenodo.org/badge/649742908.svg)](https://doi.org/10.5281/zenodo.14622040) &nbsp;
[![Wiki](https://img.shields.io/badge/Wiki-Documentation-yellow)](https://github.com/BoevaLab/UniversalEPI/wiki)


UniversalEPI is an attention-based deep ensemble designed to predict enhancer-promoter interactions up to 2 Mb, which can generalize across unseen cell types using only DNA sequence and chromatin accessibility (ATAC-seq) data as input. 

![UniversalEPI architecture](https://github.com/user-attachments/assets/28d37e3e-9e42-4df9-b726-1835f4200687)

<br/>

## Requirements

- You can install the necessary packages by creating a conda environment using the provided .yml file:
  ```bash
  conda env create -f environment.yml
  ```
  This will create an environment named "universalepi".
- Download the [data directory](https://zenodo.org/records/15079298), unzip it, and place it in the root directory such that you have `./data`.
- Download the [model checkpoints](https://doi.org/10.5281/zenodo.14622040) and place each of them in the `./checkpoints` directory. Unzip each checkpoint. 

<br/> 

## Step 1: Data Preprocessing

a. Input data processing
  - The details for processing ATAC-seq data from your raw input (BAM) or processed files (signal p-values bigwig and peaks bed) can be found in [`preprocessing/atac`](https://github.com/BoevaLab/UniversalEPI/tree/main/preprocessing/atac). This includes normalizing the bigwig and deduplication of ATAC-seq peaks.

b. Target data processing (only needed for [training and testing](https://github.com/BoevaLab/UniversalEPI?tab=readme-ov-file#universalepi-training-and-testing))
  - [`preprocessing/hic`](https://github.com/BoevaLab/UniversalEPI/tree/main/preprocessing/hic) contains the details for processing Hi-C data from your raw input (.hic or .cool) or processed files (pairwise interaction files). This includes Hi-C normalization.
  - Combine ATAC-seq and Hi-C to extract targets corresponding to ATAC peaks for each training cell line
    ```python
    python ./preprocessing/prepare_target_data.py --cell_line gm12878 --atac_bed_path ./data/atac/raw/GM12878_dedup.bed --hic_data_dir ./data/hic/
    ```
    This also saves the updated ATAC-seq peaks at `./data/atac/raw/GM12878_dedup_neg.bed` with 10% pseudopeaks added
  - Combine ATAC-seq and Hi-C to extract targets corresponding to ATAC peaks for each test cell line
    ```python
    python ./preprocessing/prepare_target_data.py --cell_line hepg2 --atac_bed_path ./data/atac/raw/HEPG2_dedup.bed --hic_data_dir ./data/hic/ --test
    ```
    The above script will run for all autosomes (chr1-22) by default. The Hi-C resolution is assumed to be 5Kb. The Hg38 genome version is considered by default. These can be modified using appropriate flags.

<br/>

## Step 2: Extract Genomic Features from Stage 1

1. Create a new config file for your cell line or condition in [`./Stage1/`](https://github.com/BoevaLab/UniversalEPI/tree/main/Stage1). See [`./Stage1/`](https://github.com/BoevaLab/UniversalEPI/tree/main/Stage1) for more details.
2. Store the genomic inputs
   ```python
   python ./Stage1/store_inputs.py --cell_line imr90
   ```
   This will store parquet files containing DNA-sequence, ATAC-seq, and mappability data at `./data/stage1_outputs/predict_imr90/`. By default, all chromosomes will be used. To use a subset of chromosomes, mention the chromosomes under "chromosome: predict:" in [`./Stage1/configs/datamodule/validation/cross_cell.yaml`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage1/configs/datamodule/validation/cross-cell.yaml).

Preprocessed data for the HepG2 cell line can be downloaded [here](https://polybox.ethz.ch/index.php/s/h98bgYpFbVquZrK).

<br/>

## Step 3: Generate Hi-C Predictions from Stage 2

1. Ensure that the atac_path (`data/stage1_outputs/`) in [`./Stage2/configs/configs.yaml`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage2/configs/configs.yaml) is correctly set. Then run
   ```python
   python ./Stage2/predict.py --config_dir ./Stage2/configs/configs.yaml --cell_line_predict imr90
   ```
   To select a subset of chromosomes for prediction, use
   ```python
   python ./Stage2/predict.py --config_dir ./Stage2/configs/configs.yaml --cell_line_predict imr90 --chroms_predict 2 6 19
   ```
   This generates `./results/imr90/paper-hg38-map-concat-stage1024-rf-lrelu-eval-stg-newsplit-newdata-atac-var-beta-neg-s1337/results.npz` which stores the following information:
    - chr (chromosome)
    - pos1 (position of ATAC-seq peak 1)
    - pos2 (position of ATAC-seq peak 2)
    - predictions (log Hi-C between peaks 1 and 2)
    - variance (aleatoric uncertainty associated with the prediction)
2. To obtain epistemic uncertainty, repeat Step 2 for each of the ten model checkpoints and take variance in predictions across the runs.

<br/>

## UniversalEPI Training and Testing

a. Train Stage1. It uses training cell lines defined in [`./Stage1/configs/datamodule/validation/cross_cell.yaml`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage1/configs/datamodule/validation/cross-cell.yaml).
  ```python
  python ./Stage1/train.py
  ```

b. Test Stage1. It uses test cell lines defined in [`./Stage1/configs/datamodule/validation/cross_cell.yaml`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage1/configs/datamodule/validation/cross-cell.yaml).
  ```python
  python ./Stage1/test.py
  ```
  
c. Train Stage2
  - Ensure that genomic data (`./data/stage1_outputs/predict_{cell_line}`) and HiC paths (`./data/hic/`) in [`./Stage2/configs/configs.yaml`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage2/configs/configs.yaml) are correct. Then run
    ```python
    python ./Stage2/main.py --config_dir ./Stage2/configs/configs.yaml --mode train
    ```
  - If npz files are already generated using [`create_dataset.py`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage2/create_dataset.py) and [`merge_dataset.py`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage2/merge_dataset.py), the data paths can be specified in [`./Stage2/configs/configs.yaml`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage2/configs/configs.yaml).


d. Test Stage2
  - Ensuring the genomic data (`./data/stage1_outputs/predict_{cell_line_test}`) and test_dir path (if exist) in [`./Stage2/configs/configs.yaml`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage2/configs/configs.yaml) are correct, run
    ```python
    python ./Stage2/main.py --config_dir ./Stage2/configs/configs.yaml --mode test
    ```
    This generates `./results/hepg2/paper-hg38-map-concat-stage1024-rf-lrelu-eval-stg-newsplit-newdata-atac-var-beta-neg-s1337/results.npz` which stores the following information:
     - chr (chromosome)
     - pos1 (position of ATAC-seq peak 1)
     - pos2 (position of ATAC-seq peak 2)
     - predictions (log Hi-C between peaks 1 and 2)
     - variance (aleatoric uncertainty associated with the prediction)
     - targets (log Hi-C)
  - `./Stage2/plot_scores.ipynb` can then be used to generate evaluation plots.


## Citing UniversalEPI

If you use UniversalEPI in your work, you can cite it using
```BibTex
@article{grover2024universalepi,
  title={UniversalEPI: harnessing attention mechanisms to decode chromatin interactions in rare and unexplored cell types},
  author={Grover, Aayush and Zhang, Lin and Muser, Till and H{\"a}fliger, Simeon and Wang, Minjia and Yates, Josephine and Van Allen, Eliezer M and Theis, Fabian J and Ibarra, Ignacio L and Krymova, Ekaterina and Boeva, Valentina},
  journal={bioRxiv},
  doi={10.1101/2024.11.22.624813},
  url={https://doi.org/10.1101/2024.11.22.624813},
  pages={2024--11},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
