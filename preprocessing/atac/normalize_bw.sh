#!/bin/bash

# Check if at least one BigWig file path is provided as an argument
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path_to_bigwig_files...>"
  exit 1
fi

multiBigwigSummary BED-file -b ../../data/atac/raw/GM12878.bigWig "$@"  -o results.npz --BED ../../data/atac/merged_ctcf.bed --outRawCounts scores_tmp.tab

sed '1s/^#//' scores_tmp.tab > scores.tab
rm scores_tmp.tab results.npz

Rscript edger_bigwig.R
rm scores.tab