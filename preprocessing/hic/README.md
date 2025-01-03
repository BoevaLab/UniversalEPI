## Hi-C Data Processing

### Step 0 (optional): Convert .hic or .cool files to pairwise interactions
- If you want to obtain the pairwise interactions from `../data/hic/HEPG2.hic` (as an example) at 5Kb resolution with ICE normalization.
  ```
  ./hic2sparse.sh ../data/hic/HEPG2.hic ../data/hic/hepg2 5000 --ice
  ```
  Remove `--ice` if the input file is already ICE-normalized. 

- Similarly, if you want to obtain the pairwise interactions from `../data/hic/HEPG2.cool` (as an example) with ICE normalization.
  ```
  ./cool2sparse.sh ../data/hic/HEPG2.cool ../data/hic/hepg2 --ice
  ```
  
- If you are interested only in a particular set of chromosomes (say, 2,6,19), the following can be used
  ```
  ./hic2sparse.sh ../data/hic/HEPG2.hic ../data/hic/hepg2 5000 --ice 2 6 19
  ```
  or
  ```
  ./cool2sparse.sh ../data/hic/HEPG2.cool ../data/hic/hepg2 --ice 2 6 19
  ```
The output pairwise interaction files (one per chromosome) will be stored in `../data/hic/hepg2/raw_iced/`. Each file will be a tab-separated file with 3 columns: pos1, pos2, hic_score.

<br/>

### Step 1: Cross-cell-type normalization
1. Ensure that the GM12878 raw_iced files are placed in `../data/hic/gm12878/raw_iced`. These will be used as reference for normalization.
2. Ensure that for all the cell lines of interest, pairwise interaction files for each chromosome are placed in `../data/hic/<cell_line>/raw_iced/chr<chrom_number>_raw.bed`
3. Apply normalization
   ```
   python normalize_hic.py --cell_lines hepg2 --data_dir ../data/hic
   ```
   By default, this script normalizes all autosomes, assumes Hi-C resolution of 5Kb, and uses `gm12878` as reference cell line. These can be modified using appropriate flags.
   
