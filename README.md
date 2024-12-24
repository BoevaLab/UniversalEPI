# UniversalEPI
UniversalEPI: Harnessing Attention Mechanisms to Decode Chromatin Interactions in Rare and Unexplored Cell Types

[![Preprint](https://img.shields.io/badge/preprint-available-green)](https://doi.org/10.1101/2024.11.22.624813) &nbsp;

<br/>

## Data Preprocessing

a. Input data processing
  - The details for processing ATAC-seq data from your raw input (BAM) or processed files (signal p-values bigwig and peaks bed) can be found in [`preprocessing/atac`](https://github.com/BoevaLab/UniversalEPI/tree/main/preprocessing/atac). This includes normalizing the bigwig and deduplication of ATAC-seq peaks.

b. Target data processing
  - [`preprocessing/hic`](https://github.com/BoevaLab/UniversalEPI/tree/main/preprocessing/hic) contains the details for processing Hi-C data from your raw input (.hic or .cool) or processed files (pairwise interaction files). This includes Hi-C normalization.
  - Combine ATAC-seq and Hi-C to extract targets corresponding to ATAC peaks for each training cell line
    ```
    python ./preprocessing/prepare_target_data.py --cell_line gm12878 --atac_bed_path ./data/atac/raw/GM12878_dedup.bed --hic_data_dir ./data/hic/
    ```
    This also saves the updated ATAC-seq peaks at `./data/atac/raw/GM12878_dedup_neg.bed` with 10% pseudopeaks added
  - Combine ATAC-seq and Hi-C to extract targets corresponding to ATAC peaks for each test cell line
    ```
    python ./preprocessing/prepare_target_data.py --cell_line hepg2 --atac_bed_path ./data/atac/raw/HEPG2_dedup.bed --hic_data_dir ./data/hic/ --test
    ```
    The above script will run for all autosomes (chr1-22) by default. The Hi-C resolution is assumed to be 5Kb. The Hg38 genome version is considered by default. These can be modified using appropriate flags.

<br/>

## Extract Genomic Features

1. Create a new config file for your cell line or condition in [`./Stage1/`](https://github.com/BoevaLab/UniversalEPI/tree/main/Stage1). See [`./Stage1/`](https://github.com/BoevaLab/UniversalEPI/tree/main/Stage1) for more details on how this can be done.
2. Store the genomic inputs
   ```
   python ./Stage1/store_inputs.py --cell_line imr90
   ```
   This will store parquet files containing DNA-sequence, ATAC-seq, and mappability data at `./data/stage1_outputs/predict_imr90/`. By default, all chromosomes will be used. To use a subset of chromosomes, mention the chromosomes under "chromosome: predict:" in [`./Stage1/configs/datamodule/validation/cross_cell.yaml`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage1/configs/datamodule/validation/cross-cell.yaml).

<br/>

## UniversalEPI Inference

1. Create the input dataset
   ```
   python ./Stage2/create_dataset.py -g ./data/stage1_outputs/predict_imr90 -s ./data/processed_data/
   ```
   This will generate `./data/processed_data/imr90_input.npz` containing information on all autosomes.

   To select a subset of chromosomes, use
   ```
   python ./Stage2/create_dataset.py -g ./data/stage1_outputs/predict_imr90 -s ./data/processed_data/ --chroms 2 6 19
   ```
2. Ensure that the test_dir path in [`./Stage2/configs/configs.yaml`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage2/configs/configs.yaml) correctly maps to `data/processed_data/imr90_input.npz`. Then run
   ```
   python ./Stage2/predict.py --config_dir ./Stage2/configs/configs.yaml
   ```
   This generates `./results/imr90/paper-hg38-map-concat-stage1024-rf-lrelu-eval-stg-newsplit-newdata-atac-var-beta-neg-s1337/results.npz` which stores the following information:
    - chr (chromosome)
    - pos1 (position of ATAC-seq peak 1)
    - pos2 (position of ATAC-seq peak 2)
    - predictions (log Hi-C between peaks 1 and 2)
    - variance (aleatoric uncertainty associated with the prediction)
3. To obtain epistemic uncertainty, repeat Step 2 for each of the ten model checkpoints and take variance in predictions across the runs.

<br/>

## UniversalEPI Training and Testing

a. Train Stage1. It uses training cell lines defined in [`./Stage1/configs/datamodule/validation/cross_cell.yaml`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage1/configs/datamodule/validation/cross-cell.yaml).
  ```
  python ./Stage1/train.py
  ```

b. Test Stage1. It uses test cell lines defined in [`./Stage1/configs/datamodule/validation/cross_cell.yaml`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage1/configs/datamodule/validation/cross-cell.yaml).
  ```
  python ./Stage1/test.py
  ```
  [Plotting scripts](https://github.com/BoevaLab/UniversalEPI/tree/main/plotting_scripts) can then be used to generate evaluation plots.
  
c. Train Stage2
  - Create input dataset for training and validation.
    ```
    python ./Stage2/create_dataset.py -g ./data/stage1_outputs/predict_gm12878 -s ./data/processed_data/ --hic_data_dir ./data/hic/ --mode train
    python ./Stage2/create_dataset.py -g ./data/stage1_outputs/predict_k562 -s ./data/processed_data/ --hic_data_dir ./data/hic/ --mode train
    python ./Stage2/create_dataset.py -g ./data/stage1_outputs/predict_gm12878 -s ./data/processed_data/ --hic_data_dir ./data/hic/ --mode val
    python ./Stage2/create_dataset.py -g ./data/stage1_outputs/predict_k562 -s ./data/processed_data/ --hic_data_dir ./data/hic/ --mode val
    ```
    This creates `gm12878_train.npz`, `k562_train.npz`, `gm12878_val.npz`, and `k562_train.npz` in `./data/processed_data`.
  - Merge training and validation cell lines
    ```
    python ./Stage2/merge_dataset.py --cell_lines gm12878 k562 --data_dir ./data/processed_data/ --phase train
    python ./Stage2/merge_dataset.py --cell_lines gm12878 k562 --data_dir ./data/processed_data/ --phase val
    ```
    This results in `train_dataset.npz` and `val_dataset.npz` in `./data/processed_data`.
  - Ensure that train and validation paths in [`./Stage2/configs/configs.yaml`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage2/configs/configs.yaml) are correct. Then run
    ```
    python ./Stage2/train.py --config_dir ./Stage2/configs/configs.yaml
    ```

d. Test Stage2
  - Create dataset for testing
    ```
    python ./Stage2/create_dataset.py -g ./data/stage1_outputs/predict_hepg2 -s ./data/processed_data/ --hic_data_dir ./data/hic/ --mode test
    ```
    This creates `./data/processed_data/hepg2_test.npz`.
  - Ensuring the test_dir path in [`./Stage2/configs/configs.yaml`](https://github.com/BoevaLab/UniversalEPI/blob/main/Stage2/configs/configs.yaml) are correct, run
    ```
    python ./Stage2/eval.py --config_dir ./Stage2/configs/configs.yaml
    ```
    This generates `./results/hepg2/paper-hg38-map-concat-stage1024-rf-lrelu-eval-stg-newsplit-newdata-atac-var-beta-neg-s1337/results.npz` which stores the following information:
     - chr (chromosome)
     - pos1 (position of ATAC-seq peak 1)
     - pos2 (position of ATAC-seq peak 2)
     - predictions (log Hi-C between peaks 1 and 2)
     - variance (aleatoric uncertainty associated with the prediction)
     - targets (log Hi-C)
  - [Plotting scripts](https://github.com/BoevaLab/UniversalEPI/tree/main/plotting_scripts) can then be used to generate evaluation plots 
