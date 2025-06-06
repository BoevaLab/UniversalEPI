#!/bin/bash

# This script converts .cool files to paired contact files
# Takes input cool file and output directory as arguments
# Optionally applies ICE normalization and processes specific chromosomes

# Usage: ./cool2sparse.sh <input_cool_file> <output_dir> [--ice] [<chromosomes>]
# Example: ./cool2sprase.sh ../data/hic/GM12878.cool .,/data/hic/gm12878 --ice 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 21 22

# Parse arguments
input_path="$1"
output_dir="$2"

# Optional arguments
ice_norm=0
chromosomes=()

# Parse optional arguments
for arg in "${@:3}"; do
    if [[ "$arg" == "--ice" ]]; then
        ice_norm="--ice"
    else
        chromosomes+=("$arg")
    fi
done

# Default chromosomes if none are provided
if [[ ${#chromosomes[@]} -eq 0 ]]; then
    chromosomes=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
fi

# Set temporary and final paths
tmp_path="$output_dir/tmp.h5"
ice_path="$output_dir/ice_corrected.h5"
ginteractions_path="$output_dir/tmp.ginteractions"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Step 1: Convert .cool to .h5
hicConvertFormat -m "$input_path" -o "$tmp_path" --inputFormat cool --outputFormat h5
echo "Converted .cool to .h5"

# Step 2: Apply ICE normalization if specified
if [[ "$ice_norm" == "--ice" ]]; then
    hicCorrectMatrix correct --matrix "$tmp_path" --filterThreshold -1.1 4.5 -o "$ice_path" --perchr --chromosomes "${chromosomes[@]}"
    tmp_path="$ice_path" # Update tmp_path to point to ICE-corrected file
    echo "Applied ICE normalization"
fi

# Step 3: Convert h5 to ginteractions
hicConvertFormat -m "$tmp_path" -o "$ginteractions_path" --inputFormat h5 --outputFormat ginteractions
tmp_out_path="$ginteractions_path.tsv"
echo "Converted .h5 to .ginteractions"

# Clean up temporary files
rm "$tmp_path"

# Step 4: Convert ginteractions to sparse format using Python script
python ./tsv2sparse.py --input_file "$tmp_out_path" --output_dir "$output_dir" --chroms "${chromosomes[@]}"

# Clean up temporary files
rm $tmp_out_path

echo "Conversion completed. Output saved to: $output_dir"
