import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
from scipy.interpolate import splrep, BSpline

import matplotlib.pyplot as plt
import seaborn as sns

import os
from tqdm import tqdm


def get_percentiles(cell, data_dir, chrom=range(1,23), bin_size=5000, len_bins=401, reference=None, save=False, smooth_map=None):
    before_norm = {}
    output = {}
    bool_ref = True if reference else False

    if save:
        os.makedirs(data_dir + "/{}/iced_quan_norm/".format(cell), exist_ok=True)
        
    print(f'Cell line: {cell}; Reference: {bool_ref}')
    for chr_n in tqdm(chrom):
        kr_norm_path = os.path.join(data_dir, cell, "raw_iced", f"chr{chr_n}_raw.bed")
        try:
            hic_kr = pd.read_csv(kr_norm_path, sep="\t", names=["bin1", "bin2", "score"])
        except:
            raise FileNotFoundError(f"File not found: {kr_norm_path}")
        hic_kr["dist"] = hic_kr["bin2"] - hic_kr["bin1"]
        hic_kr = hic_kr[hic_kr["dist"]<=(len_bins*bin_size)].reset_index(drop=True)
        
        df = pd.DataFrame({
            'score': hic_kr['score'].tolist(), 
            'dist': hic_kr['dist'].tolist()
            })
        df_before = df.copy()
        before_norm[chr_n] = df_before
        
        df_copy = df.copy()
        out_dfs = []
        save_dfs = []
        if bool_ref:
            for i in range(len_bins+1):
                df_strat = df_copy[(df_copy.dist == (i*bin_size))]
                df_dist = df_strat.loc[:, ['dist']]
                df_score = df_strat.score.to_numpy()
                if smooth_map and (i > 0):
                    med = smooth_map[cell][chr_n]['med'][i-1]
                    mad = smooth_map[cell][chr_n]['mad'][i-1]
                else:
                    med = np.median(df_score)
                    mad = median_abs_deviation(df_score)
                z_score = (df_score - med)/mad
                score = (z_score*reference[chr_n][i][1])+reference[chr_n][i][0]
                df_dist['score'] = score
                out_df = df_dist
                out_dfs.append(out_df)
                if save:
                    hic = hic_kr[(hic_kr.dist == (i*bin_size))]
                    df_meta = hic.loc[:,['bin1', 'bin2']]
                    df_meta['score'] = score
                    save_df = df_meta
                    save_dfs.append(save_df)
            if save:
                assert len(save_dfs) > 0, "something is wrong. Distance stratified df creation failed."
                save_df = pd.concat(save_dfs)
                save_df = save_df[save_df.score > 0]
                save_path = os.path.join(data_dir, cell, "iced_quan_norm", f"chr{chr_n}_robust_norm.bed")
                save_df.to_csv(save_path, index=None)
            out_df = pd.concat(out_dfs)
            output[chr_n] = out_df
        else:
            output[chr_n] = {}
            for i in range(len_bins+1):
                df_strat = df_copy[(df_copy.dist == (i*bin_size))]
                df_score = df_strat.score.to_numpy()
                if smooth_map:
                    med = smooth_map[cell][chr_n]['med'][i]
                    mad = smooth_map[cell][chr_n]['mad'][i]
                else:
                    med = np.median(df_score)
                    mad = median_abs_deviation(df_score)
                output[chr_n][i] = (med, mad)
    
    return before_norm, output

def _plot(mapping, cells, data_dir, chrom=range(1,23), before_norm=True, len_bins=201, bin_size=5000):
    os.makedirs(os.path.join(data_dir, "plots"), exist_ok=True)

    for chr_n in chrom:
        cell_line = [] 
        score = []
        dist = []
        for cell in cells:
            try:
                df = (mapping[cell])[chr_n]
            except:
                continue
            for i in range(1, len_bins+1):
                strat_scores = (df[df["dist"] == i*bin_size]["score"]).mean()
                dist.append(np.log(i*bin_size))
                score.append(np.log(strat_scores+1))
                cell_line.append(cell)
        
        if len(cell_line) == 0:
            continue
        
        plot_df = pd.DataFrame({
            'Cell Line': cell_line,
            'Mean Log. HiC': score,
            'Log. Distance': dist
        })
        plot_df['Cell Line'] = plot_df['Cell Line'].apply(lambda x: x.upper())

        line_plot = sns.lineplot(x="Log. Distance", y="Mean Log. HiC", data=plot_df, hue="Cell Line").set_title(f'Chr{chr_n}')
        if before_norm:
            save_path = os.path.join(data_dir, "plots", f"before_norm_robust_chr{chr_n}.svg")
        else:
            save_path = os.path.join(data_dir, "plots", f"after_norm_robust_chr{chr_n}.svg")
        line_plot.figure.savefig(save_path, dpi=200, transparent=True)
        plt.clf()


def normalize_hic(cells, data_dir, reference, bin_size=5000, len_bins=201, chrom=range(1,23), smooth_map=None, plot=False):
    if plot:
        before_map = {}
        after_map = {}
        for cell_line in cells:
            before_norm, after_norm = get_percentiles(cell_line, data_dir, chrom=chrom, bin_size=bin_size, len_bins=len_bins, reference=reference,
                                                    save=True, smooth_map=smooth_map)
            before_map[cell_line] = before_norm
            after_map[cell_line] = after_norm

        print('Plotting before normalization')
        _plot(before_map, cells, data_dir, chrom=chrom, before_norm=True, len_bins=len_bins, bin_size=bin_size)
        print('Plotting after normalization')
        _plot(after_map, cells, data_dir, chrom=chrom, before_norm=False, len_bins=len_bins, bin_size=bin_size)
    else:
        for cell_line in cells:
            get_percentiles(cell_line, data_dir, chrom=chrom, bin_size=bin_size, len_bins=len_bins, reference=reference, save=True, smooth_map=smooth_map)


def get_smooth_map(cell_lines, data_dir, chrom, bin_size, len_bins):
    smooth_map = {}
    dist_list = [np.log((i*bin_size)+1) for i in range(1,len_bins+1)]
    for cell in cell_lines:
        smooth_map[cell] = {}
        for chr_n in tqdm(chrom):

            kr_norm_path = data_dir + "/{}/raw_iced/chr{}_raw.bed".format(cell, chr_n)
            try:
                hic_kr = pd.read_csv(kr_norm_path, sep="\t", names=["bin1", "bin2", "score"])
            except:
                continue
            hic_kr["dist"] = hic_kr["bin2"] - hic_kr["bin1"]
            hic_kr = hic_kr[hic_kr["dist"]<=(len_bins*bin_size)].reset_index(drop=True)
            df = pd.DataFrame({
                'score': hic_kr['score'].tolist(), 
                'dist': hic_kr['dist'].tolist()
            })
            med_list = []
            mad_list = []
            for i in range(1, len_bins+1):
                df_strat = df[(df.dist == (i*bin_size))]
                df_score = df_strat.score.to_numpy()
                med = np.median(df_score)
                mad = median_abs_deviation(df_score)
                med_list.append(med)
                mad_list.append(mad)
            tck_med = splrep(dist_list, med_list, s=5)
            tck_mad = splrep(dist_list, mad_list, s=5)

            
            smooth_map[cell][chr_n] = {
                'med': [*BSpline(*tck_med)(dist_list)], 
                'mad': [*BSpline(*tck_mad)(dist_list)]
            }
    return smooth_map