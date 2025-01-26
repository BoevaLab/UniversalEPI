## ATAC-seq Data Processing

### Cross-cell-type normalization
Ensure that GM12878 ATAC-seq bigwig, which will be used as the reference, is present in `../../data/atac/raw`.

- If you have bam files as input
   ```
   python normalize_atac.py -p ../../data/atac/raw/ --input_bam ../../data/atac/raw/IMR90.bam ../../data/atac/raw/HepG2.bam 
   ```
- If you have bigwig and peak files as input
   ```
   python normalize_atac.py -p ../../data/atac/raw/ --input_bw ../../data/atac/raw/IMR90.bigWig ../../data/atac/raw/HepG2.bigWig --input_bed ../../data/atac/raw/IMR90.bed ../../data/atac/raw/HepG2.bed
   ```


This will create the normalized bigwig files `../data/atac/raw/IMR90_normalized.bw`, `../data/atac/raw/HepG2_normalized.bw` and deduplicated peak files `../data/atac/raw/IMR90_dedup.bed`, `../data/atac/raw/HepG2_dedup.bed`.
