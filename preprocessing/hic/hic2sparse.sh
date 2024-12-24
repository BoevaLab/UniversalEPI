#!/bin/bash

# This script converts .hic files to paired contact files
# Takes input hic file and output directory as arguments
# Optionally applies ICE normalization and processes specific chromosomes

# Usage: ./hic2sparse.sh <input_hic_file> <output_dir> <resolution> [--ice] [<chromosomes>]
# Example: ./hic2sparse.sh ../data/hic/GM12878.hic ../data/hic/gm12878 5000 --ice 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 21 22

# Parse arguments
input_path="$1"
output_dir="$2"
resolution="$3"

# Optional arguments
ice_norm=0
chromosomes=()

# Parse optional arguments
for arg in "${@:4}"; do
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
tmp_cool_path="$output_dir/tmp.cool"
tmp_path="$output_dir/tmp.h5"
ice_path="$output_dir/ice_corrected.h5"
ginteractions_path="$output_dir/tmp.ginteractions"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Step 0: Convert .hic to .cool
hicConvertFormat -m "$input_path" -o "$tmp_cool_path" --inputFormat hic --outputFormat cool --resolutions "$resolution"
echo "Converted .hic to .cool"

# Step 1: Convert .cool to .h5
hicConvertFormat -m "$tmp_cool_path" -o "$tmp_path" --inputFormat cool --outputFormat h5
echo "Converted .cool to .h5"

# Clean up temporary files
rm "$tmp_cool_path"

# Step 2: Apply ICE normalization if specified
if [[ "$ice_norm" == "--ice" ]]; then
    hicCorrectMatrix correct --matrix "$tmp_path" --filterThreshold -1.1 4.5 -o "$ice_path" --perchr --chromosomes "${chromosomes[@]}"
    rm "$tmp_path" # Remove uncorrected file
    tmp_path="$ice_path" # Update tmp_path to point to ICE-corrected file
    echo "Applied ICE normalization"
fi

# Step 3: Convert h5 to ginteractions
hicConvertFormat -m "$tmp_path" -o "$ginteractions_path" --inputFormat h5 --outputFormat ginteractions
echo "Converted .h5 to .ginteractions"

# Clean up temporary files
rm "$tmp_path"

# Step 4: Convert ginteractions to sparse format using Python script
python ./tsv2sparse.py "$ginteractions_path" "$output_dir" "${chromosomes[@]}"

# Clean up temporary files
rm "$gineractions_path"

echo "Conversion completed. Output saved to: $output_dir/raw_iced"
