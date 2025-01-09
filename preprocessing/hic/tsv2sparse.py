import os
import argparse
import pandas as pd

def iced2sparse(iced_file, output_dir, chroms):
    hic = pd.read_csv(iced_file, sep='\t', header=None, names=['chr', 'bin1', 's1', 's2', 'bin2', 's3', 'score'])
    hic = hic[['chr', 'bin1', 'bin2', 'score']]
    for chrom in chroms:
        result = hic[hic['chr'] == int(chrom)]
        if 'chr' in chrom:
            chr_str = chrom
        else:
            chr_str = 'chr' + chrom
        if len(result) == 0:
            result = hic[hic['chr'] == chr_str]
        df = result[['bin1', 'bin2', 'score']]
        os.makedirs(os.path.join(output_dir, 'raw_iced'), exist_ok=True)
        out_path = os.path.join(output_dir, 'raw_iced', f'{chr_str}_raw.bed')
        df.to_csv(out_path, sep='\t', index=None, header=None)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert ginteractions tsv file to sparse matrix')
    parser.add_argument('--input_file', type=str, help='Path to tsv file')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    parser.add_argument('--chroms', type=str, nargs="+", help='Chromosomes to process')
    return parser.parse_args()

def main():
    args = parse_args()
    chroms = args.chroms
    iced2sparse(args.input_file, args.output_dir, chroms)

if __name__ == '__main__':
    main()
