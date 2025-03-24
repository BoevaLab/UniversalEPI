import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import pyBigWig

from bam_to_bw_pval import bam_to_bw, run_shell_cmd, rm_f

def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--out_path', required=True, help='Output path')
    parser.add_argument('--input_bw', nargs='+', help='List of input bigwig files', default=[])
    parser.add_argument('--input_bed', nargs='+', help='List of input bed files', default=[])
    parser.add_argument('--input_bam', nargs='+', help='List of input bam files', default=[])
    parser.add_argument('--chrom_sizes', help='Chromosome sizes file', default='../../data/genome/hg38.ucsc.sizes')
    return parser.parse_args()

def _input_exists(input_files):
    if len(input_files) == 0:
        return False
    return True

def _scale_bigwig(input_bigwig_file, output_bigwig_file, scale_factor, precision=3):
    input_bw = pyBigWig.open(input_bigwig_file)
    output_bw = pyBigWig.open(output_bigwig_file, "w")

    output_bw.addHeader(list(input_bw.chroms().items()))
    chroms = input_bw.chroms()

    chunk_size = 100000

    for chrom in chroms:
        chrom_size = chroms[chrom]
        for start in tqdm(range(0, chrom_size, chunk_size), desc=f'{chrom}'):
            end = min(start + chunk_size, chrom_size)
            values = input_bw.values(chrom, start, end)
            values = np.nan_to_num(values) * scale_factor
            values = np.round(values, precision)
            starts = np.arange(start, end)
            ends = starts + 1
            output_bw.addEntries([chrom]*len(starts), starts.tolist(), ends=ends.tolist(), values=values.tolist())

    input_bw.close()
    output_bw.close()


def _dedup_bed(file_path, out_path):
    save_path = os.path.join(out_path, os.path.basename(file_path).split('.')[0]+'_dedup.bed')

    names = ['chrom', 'start', 'end', 'name', 'signal', 'strand', 'score', 'p', 'q', 'peak']
    select_names = ['chrom', 'start', 'score', 'peak']
    df_full = pd.read_csv(file_path, sep='\t', names=names)
    df = df_full[select_names]
    ind_list = []
    for chr_n in range(1, 23):
        chr = f"chr{chr_n}"
        df_chr = df[df.chrom == chr]
        df_chr.sort_values(['start', 'score'], ascending=[True, False], inplace=True)
        starts = sorted(list(set(df_chr['start'].values.tolist())))
        for start in tqdm(starts, desc=f'{chr}'):
            df_start = df_chr[df_chr.start==start]
            if len(df_start) == 1:
                i = list(df_start.index.values)[0]
                ind_list.append(i)
                continue
            while True:
                row =  df_start.head(1)
                i = list(row.index.values)[0]
                max_peak = list(row.peak.values)[0]
                ind_list.append(i)
                df_start = df_start[abs(df_start.peak-max_peak)>500]
                if len(df_start) == 0:
                    break
    new_df = df_full.loc[ind_list]
    new_df.to_csv(save_path, sep='\t', index=None, header=False)
    print(f'Deduplicated peaks saved at {save_path}')

def _get_prefix(file_path):
    return os.path.basename(file_path).split('.')[0]


def main():
    args = _parse()
    
    # Check if input bam files are provided. Otherwise bed and bigwig files should be provided
    if _input_exists(args.input_bam) and (_input_exists(args.input_bed) or _input_exists(args.input_bw)):
        raise ValueError('Provide either input bam files or input bed and bigwig files')
    elif not _input_exists(args.input_bam):
        if not _input_exists(args.input_bed) or not _input_exists(args.input_bw):
            raise ValueError('Provide both input bed and bigwig files')
        if len(args.input_bed) != len(args.input_bw):
            raise ValueError('Number of input bed and bigwig files should be the same')
    bigwig_files = args.input_bw
    bed_files = args.input_bed
    
    # Convert bam to bed and bigwig
    if _input_exists(args.input_bam):
        bigwig_files = []
        bed_files = []
        for bam_file in args.input_bam:
            print(f'Converting {bam_file} to bed and bigwig')
            out_prefix = os.path.join(args.out_path, _get_prefix(bam_file))
            bam_to_bw(bam_file, out_prefix, args.chrom_sizes)
            bigwig_files.append(f'{out_prefix}.pval.signal.bigwig')
            bed_files.append(f'{out_prefix}.peaks.narrowpeak')
    
    # Remove GM12878 bigwig file
    for bw in bigwig_files:
        if 'GM12878.bigWig' in bw:
            bigwig_files.remove(bw)
            break
    
    # Normalize bigwig files
    run_shell_cmd(f'./normalize_bw.sh {" ".join(bigwig_files)}')

    df = pd.read_csv('normFactors.txt', sep='\t')
    scale_factors = df['NormFactor'].values
    for i, bigwig_file in enumerate(bigwig_files):
        out_bigwig = os.path.join(args.out_path, f'{_get_prefix(bigwig_file)}_normalized.bw')
        print(f'Scaling {bigwig_file} by {scale_factors[i]}')
        _scale_bigwig(bigwig_file, out_bigwig, scale_factors[i])

    # Remove temporary files
    rm_f('normFactors.txt')
    
    # Deduplicate bed files
    for bed_file in bed_files:
        _dedup_bed(bed_file, args.out_path)

if __name__=="__main__":
    main()
