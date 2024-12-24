#!/bin/bash

#SBATCH --job-name=cNMF4-data
#SBATCH --output=/cluster/work/boeva/tacisu/trepi/job_output/hg38-data-cNMF4.out.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --time=10:00:00
 
python ~/mutiger-explore/EPI-TREPI/store_meta_data_all.py --cell_line cNMF4 --save_path /cluster/work/boeva/tacisu/trepi
