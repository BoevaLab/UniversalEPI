## Hi-C Data Processing

### Step 0 (optional): Convert .hic or .cool files to pairwise interactions
- If you want to obtain the pairwise interactions from `./data/IMR90.hic` (as an example) at 5Kb resolution with ICE normalization.
  ```
  ./hic2sparse.sh ./data/hic/IMR90.hic ./data/imr90 5000 --ice
  ```
  Remove `--ice` if the input file is already ICE-normalized. 

- Similarly, if you want to obtain the pairwise interactions from `./data/IMR90.cool` (as an example) with ICE normalization.
  ```
  ./cool2sparse.sh ./data/hic/IMR90.cool ./data/imr90 --ice
  ```
  
- If you are interested only in a particular set of chromosomes (say, 2,6,19), the following can be used
  ```
  ./hic2sparse.sh ./data/hic/IMR90.hic ./data/imr90 5000 --ice 2 6 19
  ```
  or
  ```
  ./cool2sparse.sh ./data/hic/IMR90.cool ./data/imr90 --ice 2 6 19
  ```
The output pairwise interaction files (one per chromosome) will be stored in `./data/imr90/raw_iced/`. Each file will be a tab-separated file with 3 columns: pos1, pos2, hic_score.

<br/>

### Step 1: Cross-cell-type normalization
1. Download the GM12878 raw_iced files and place them in `./data/gm12878/raw_iced`. These will be used as reference for normalization.
  ```
  wget TBD
  ```
2. Ensure that for all the cell lines of interest, pairwise interaction files for each chromosome are placed in `./data/<cell_line>/raw_iced/chr<chrom_number>_raw.bed`
3. Apply normalization
   ```
   python normalize_hic.py --cell_lines imr90 --data_dir ./data/
   ```
   By default, this script normalizes all autosomes, assumes Hi-C resolution of 5Kb, and uses `gm12878` as reference cell line. These can be modified using appropriate flags.
   
