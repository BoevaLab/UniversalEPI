import os
import argparse
import random
import pickle as pkl
from tqdm import tqdm

import pandas as pd
import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import coo_matrix

from utils import get_chrom_size, get_df, gaussian_kernel, get_df_from_mat

def smooth_hic(resolution, data_dir, chroms, genome="hg38"):
    window_size = int(20e6//resolution)
    step_size = int(18e6//resolution)

    if resolution == 5000:
        thresholds = [t//resolution for t in [25000, 100000, 250000]]
    elif resolution == 10000:
        thresholds = [t//resolution for t in [50000, 200000, 500000]]
    else:
        raise ValueError("Invalid resolution. Please choose either 5000 or 10000.")
    
    kernels = [gaussian_kernel(5, sigma=s) for s in [0.5, 0.65, 0.8]]
    chrom_sizes = get_chrom_size(genome)
    for chr_n in chroms:
        hic_path = os.path.join(data_dir, "iced_quan_norm", f"chr{chr_n}_robust_norm.bed")
        hic = pd.read_csv(hic_path)
        hic['bin1'] = hic['bin1']//resolution
        hic['bin2'] = hic['bin2']//resolution

        max_len = int(chrom_sizes[chr_n]//resolution)
        df_list = []
        for i in tqdm(range(0, max_len, step_size), desc=f"Smoothing chr{chr_n}"):
            df = get_df(hic, i, i, window_size)
            if df is None:
                continue

            rows = df.bin1.values
            cols = df.bin2.values
            values = df.score.values
            rows = rows-i
            cols = cols-i

            sparse_mat = coo_matrix((values,(rows,cols)), shape=(window_size, window_size))
            dense_mat = sparse_mat.toarray()

            smooth_mat = np.copy(dense_mat)
            n = len(smooth_mat)

            prev_threshold = 0
            for kernel, threshold in zip(kernels, thresholds):
                smooth_submat = convolve2d(dense_mat, kernel, mode='same')
                for idx in range(n):
                    start = min(0, idx-prev_threshold-threshold)
                    end = idx-prev_threshold-1
                    if end > 0:
                        smooth_mat[idx, start:end] = smooth_submat[idx, start:end]
                    start = idx+prev_threshold
                    end = max(n+1, idx+prev_threshold+threshold+1)
                    if start < n:
                        smooth_mat[idx, start:end] = smooth_submat[idx, start:end]
                prev_threshold = threshold

            kernel = gaussian_kernel(5, sigma=1)
            smooth_submat = convolve2d(dense_mat, kernel, mode='same')
            threshold = n-1
            for idx in range(n):
                start = min(0, idx-prev_threshold-threshold)
                end = idx-prev_threshold-1
                if end > 0:
                    smooth_mat[idx, start:end] = smooth_submat[idx, start:end]
                start = idx+prev_threshold
                end = max(n+1, idx+prev_threshold+threshold+1)
                if start < n:
                    smooth_mat[idx, start:end] = smooth_submat[idx, start:end]

            smooth_mat = coo_matrix(smooth_mat)

            df_list.append(get_df_from_mat(smooth_mat, step=i))

        df = pd.concat(df_list, ignore_index=True)
        df = df[df.score > 0]
        df.sort_values(by='score', ascending=False, inplace=True, ignore_index=True)
        df = df.drop_duplicates(subset=['bin1', 'bin2'])
        df.sort_values(by=['bin1','bin2'], ascending=True, inplace=True, ignore_index=True)
        
        os.makedirs(os.path.join(data_dir, "smooth_hic"), exist_ok=True)
        save_path = os.path.join(data_dir, "smooth_hic", f"chr{chr_n}_robust_norm.bed")    
        df.to_csv(save_path, index=None)

def is_valid_pos(pos, df):
    if len(df[abs(df['loc']-pos)<500]) > 0:
        return False
    return True

def add_neg_samples(atac_bed_path, chroms, negative_sampling_rate=0.1):
    names = ['chrom', 'start', 'end', 'name', 'signal', 'strand', 'score', 'p', 'q', 'peak']
    df = pd.read_csv(atac_bed_path, sep='\t', names=names)
    df['loc'] = df['start'] + df['peak']
    df_list = []
    for chr_n in chroms:
        chr = f"chr{chr_n}"
        df_chr = df[df.chrom == chr]
        n = int(len(df_chr)*negative_sampling_rate)
        df_chr.sort_values(['loc', 'score'], ascending=[True, False], inplace=True, ignore_index=True)
        start, end = df_chr.iloc[0]['loc'], df_chr.iloc[-1]['loc']
        added = 0
        while added <= n:
            pos = random.randint(start,end)
            if is_valid_pos(pos, df_chr):
                row = [chr, pos-250, pos+250, f'Peak_neg_{added}', 500, '.', 1.0, 1.0, 1.0, 250, pos]
                df_chr = pd.concat([pd.DataFrame([row], columns=df_chr.columns), df_chr])
                added += 1
        df_list.append(df_chr)
    new_df = pd.concat(df_list, ignore_index=True)
    new_df.sort_values(['chrom', 'start', 'score'], ascending=[True, True, False], inplace=True, ignore_index=True)
    new_df = new_df[names]
    return new_df

def save_target_data(full_atac_seq, hic_data_dir, bin_size, len_bins, chroms):
    for chr_n in chroms:
        hic = pd.read_csv(os.path.join(hic_data_dir, f"chr{chr_n}_robust_norm.bed"), sep='\t', names=['bin1', 'bin2', 'score'])

        atac_seq = full_atac_seq[full_atac_seq["chrom"] == "chr{}".format(chr_n)]
        atac_seq.loc[:, "hic_start"] = (np.floor((atac_seq["start"] + atac_seq["peak"]) / bin_size) * bin_size).astype(int)
        atac_seq = atac_seq.sort_values(by=["hic_start"], ascending = True)
        
        peak_dict_list = []
        
        first_row = True
        for _, atac_row in tqdm(atac_seq.iterrows(), total=len(atac_seq), desc="Calculating interaction scores for chr{}".format(chr_n)):
            peak_name = atac_row["name"]
            if (first_row) or (atac_row["hic_start"] != start_int) or ((atac_row["hic_start"] + (len_bins*bin_size)) != end_int):
                start_int = atac_row["hic_start"]
                end_int = atac_row["hic_start"] + (len_bins*bin_size)
                rel_rows = atac_seq[(atac_seq["hic_start"] >= start_int) & (atac_seq["hic_start"] <= end_int)]
                rel_hic = hic[hic["bin1"] == start_int]
                first_row = False

            for _, rel_row in rel_rows.iterrows():
                peak_2_name = rel_row["name"]
                this_hic = rel_hic[(rel_row["hic_start"] == rel_hic["bin2"])]
                if len(this_hic) > 0:
                    hic_perc = max(this_hic["score"])
                else:
                    hic_perc = 0
                dist = abs(rel_row["hic_start"]-start_int)
                dist = int(dist/bin_size)
                peak_dict_list.append({"peak_1": peak_name, "peak_2": peak_2_name, "interaction": hic_perc, "dist": dist})
                    
        interactions = pd.DataFrame(peak_dict_list)
        flank_len = 200

        atac_seq['loc'] = atac_seq['start'] + atac_seq['peak']
        atac_seq = atac_seq.sort_values(by='loc', ascending=True)
        atac_list = list(atac_seq["name"])
        atac_list = list(map(str, atac_list))

        int_matrix = np.zeros((len(atac_list), len(atac_list)))

        for _, inter_row in tqdm(interactions.iterrows(), total=len(interactions), desc="Creating interaction matrix for chr{}".format(chr_n)):
            try:
                idx_1 = atac_list.index(str(inter_row["peak_1"]))
            except:
                idx_1 = atac_list.index(str(int(inter_row["peak_1"])))
            try:
                idx_2 = atac_list.index(str(inter_row["peak_2"]))
            except:
                idx_2 = atac_list.index(str(int(inter_row["peak_2"])))
            int_matrix[idx_1][idx_2]= inter_row["interaction"]
            int_matrix[idx_2][idx_1]= inter_row["interaction"]

        interaction_list = []    
        for p_n in range (flank_len, len(atac_list)-flank_len, 1):
            target = []
            i,j = p_n,p_n
            min_index = p_n-flank_len
            while True:
                target.append(int_matrix[i][j])
                i -= 1
                if i >= min_index:
                    target.append(int_matrix[i][j])
                else:
                    break
                j += 1
            interaction_list.append(np.array(target))

        np_int_list = np.array(interaction_list)
        
        base_dir = os.path.dirname(hic_data_dir)
        os.makedirs(os.path.join(base_dir, "target_data"), exist_ok=True)
        output_path = os.path.join(base_dir, "target_data", f"chr{chr_n}_target.pkl")
        with open(output_path, 'wb') as f:
            pkl.dump(np_int_list, f)

def parse_args():
    parser = argparse.ArgumentParser(description="Smooth Hi-C data")
    parser.add_argument("--cell_line", type=str, help="Cell line", required=True)
    parser.add_argument("--atac_bed_path", type=str, help="Path to the ATAC bed file", required=True)
    parser.add_argument("--resolution", default=5000, type=int, help="Resolution")
    parser.add_argument("--hic_data_dir", default='./hic/data/', type=str, help="Path to the directory containing Hi-C data")
    parser.add_argument("--chrom", default=[*range(1,23)], type=int, nargs="+", help="Chromosomes to consider")
    parser.add_argument("--max_len", default=2_000_000, type=int, help="Maximum length of input")
    parser.add_argument("--genome", default='hg38', type=str, help="Genome")
    parser.add_argument("--test", action="store_true", help="Preprocess test data")

    return parser.parse_args()

def main():
    args = parse_args()
    cell_line = args.cell_line
    bin_size = args.resolution
    atac_bed_path = args.atac_bed_path
    chroms = args.chrom
    genome = args.genome
    is_test = args.test
    len_bins = (args.max_len//bin_size)+1
    base_dir = os.path.join(args.hic_data_dir, cell_line)

    if is_test:
        smooth_hic(bin_size, base_dir, chroms, genome)
        names = ['chrom', 'start', 'end', 'name', 'signal', 'strand', 'score', 'p', 'q', 'peak']
        atac_bed = pd.read_csv(atac_bed_path, sep='\t', names=names)
        hic_data_dir = os.path.join(base_dir, "smooth_hic")
    else:
        atac_bed = add_neg_samples(atac_bed_path, chroms)
        atac_save_path = atac_bed_path.replace(".bed", "_neg.bed")
        atac_bed.to_csv(atac_save_path, sep='\t', index=False, header=False)
        hic_data_dir = os.path.join(base_dir, "iced_quan_norm")
    
    save_target_data(atac_bed, hic_data_dir, bin_size, len_bins, chroms)

if __name__ == "__main__":
    main()