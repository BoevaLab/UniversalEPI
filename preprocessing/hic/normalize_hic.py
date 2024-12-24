import argparse

from utils import get_percentiles, normalize_hic, get_smooth_map

def robust_norm(cell_lines, data_dir, ref_dir=None, chrom=range(1,23), bin_size=5000, len_bins=401, reference="gm12878", plot=False):
    print('Applying cross-cell line robust z-score normalization...')
    if reference not in cell_lines:
        cell_lines = [reference] + cell_lines
        ref_in_cell = False
    else:
        ref_in_cell = True

    if ref_dir is None:
        ref_dir = data_dir
    
    print('Fitting splines...')
    smooth_map = get_smooth_map(cell_lines, data_dir, chrom, bin_size, len_bins)
    
    # get percentiles of the reference cell line
    _, reference_perc = get_percentiles(reference, ref_dir, chrom=chrom, bin_size=bin_size, len_bins=len_bins)

    # z-score normalization on all cell lines based on the reference cell line
    if ref_in_cell:
        normalize_hic(cell_lines, data_dir, reference_perc, bin_size=bin_size, len_bins=len_bins, chrom=chrom, smooth_map=smooth_map, plot=plot)
    else:
        cell_lines = cell_lines[1:]
        normalize_hic(cell_lines, data_dir, reference_perc, bin_size=bin_size, len_bins=len_bins, chrom=chrom, smooth_map=smooth_map, plot=plot)
    print('Cross-cell line normalization done!')

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Hi-C data")
    parser.add_argument("--cell_lines", type=str, nargs="+", help="List of cell lines", required=True)
    parser.add_argument("--data_dir", default='../data/hic', type=str, help="Path to the directory containing Hi-C data")
    parser.add_argument("--chrom", default=[*range(1,23)], type=int, nargs="+", help="Chromosomes to consider")
    parser.add_argument("--bin_size", default=5000, type=int, help="Bin size")
    parser.add_argument("--max_len", default=2_000_000, type=int, help="Number of bins")
    parser.add_argument("--reference", default='gm12878', type=str, help="Reference cell line")
    parser.add_argument("--plot", action="store_true", help="Plot the before and after normalization maps")
    return parser.parse_args()

def main():
    args = parse_args()
    cell_lines = args.cell_lines
    data_dir = args.data_dir
    ref_dir = "../data/hic/"
    chroms = args.chrom
    bin_size = args.bin_size
    len_bins = (args.max_len//bin_size)+1
    reference = args.reference
    plot = args.plot

    robust_norm(cell_lines, data_dir, ref_dir=ref_dir, chrom=chroms, bin_size=bin_size, len_bins=len_bins, reference=reference, plot=plot)

if __name__ == "__main__":
    main()