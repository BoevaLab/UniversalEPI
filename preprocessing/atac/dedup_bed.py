import os
from tqdm import tqdm
import argparse
import pandas as pd


def deduplicate_bed(file_path, chroms, save_path):
    names = ['chrom', 'start', 'end', 'name', 'signal', 'strand', 'score', 'p', 'q', 'peak']
    select_names = ['chrom', 'start', 'score', 'peak']
    df_full = pd.read_csv(file_path, sep='\t', names=names)
    df = df_full[select_names]
    ind_list = []
    for chr_n in chroms:
        chr = f"chr{chr_n}"
        df_chr = df[df.chrom == chr]
        df_chr.sort_values(['start', 'score'], ascending=[True, False], inplace=True)
        starts = sorted(list(set(df_chr['start'].values.tolist())))
        for start in tqdm(starts):
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

def parse_args():
    parser = argparse.ArgumentParser(description="Deduplicate ATAC-seq peaks")
    parser.add_argument('-f',"--file_path", type=str, help="Path to the bed file containing ATAC-seq peaks", required=True)
    parser.add_argument('-c', "--chroms", default=[*range(1,23)], type=int, nargs="+", help="Chromosomes to consider")
    return parser.parse_args()

def main():
    args = parse_args()
    file_path = args.file_path
    save_path = os.path.dirname(file_path)
    save_path = os.path.join(save_path, os.path.basename(file_path).split('.')[0]+'_dedup.bed')
    chroms = args.chroms
    
    deduplicate_bed(file_path, chroms, save_path)

if __name__ == "__main__":
    main()