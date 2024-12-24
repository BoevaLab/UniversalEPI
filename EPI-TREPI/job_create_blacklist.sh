#!/bin/bash

#SBATCH --job-name=hg38_blacklist
#SBATCH --output=/cluster/work/boeva/tacisu/trepi/job_output/hg38-data-swap-blacklist.out.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --time=10:00:00
 
python ~/mutiger-explore/EPI-TREPI/create_blacklist.py
