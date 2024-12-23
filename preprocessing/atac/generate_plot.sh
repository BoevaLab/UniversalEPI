#!/bin/bash

computeMatrix reference-point --referencePoint center -b 2500 -a 2500 -R merged_ctcf.bed -S ../../data/atac/*.bigWig --skipZeros -o ./matrix.gz

plotProfile -m matrix.gz -out before_norm.svg --plotTitle "Before Normalization" --refPointLabel "Center" --regionsLabel " " --perGroup --dpi 200
rm matrix.gz
