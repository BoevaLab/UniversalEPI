#!/bin/bash

#SBATCH --job-name=hg38_hepg2
#SBATCH --output=/cluster/work/boeva/tacisu/trepi/job_output/hg38-data-hepg2.out.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=450G
#SBATCH --time=10:00:00
 
#python ~/mutiger-explore/EPI-TREPI/store_meta_data_hg38_dnase_seq.py --cell_line hepg2_merged --save_path /cluster/work/boeva/tacisu/trepi --infer
python ~/mutiger-explore/EPI-TREPI/store_meta_data_hg38_dnase_seq.py --cell_line hepg2 --save_path /cluster/work/boeva/tacisu/trepi --infer
