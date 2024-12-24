import torch

import pandas as pd
import numpy as np
import gc

import argparse


# python store_meta_data_all.py --cell_line gm12878 --save_path /mydata/mutiger/shared/hg38_new_split/

def filter_sort_atac(atac_seq, chr_str):
    '''
    takes pandas.DataFrame of all atac-seq peaks and filters for one
    chromosome as well as sorts them ascending by start nucleotide
    '''
    try:
        atac_seq_chr = atac_seq[atac_seq["chrom"] == chr_str].drop(["sth2", "sth3"], axis=1)
    except:
        atac_seq_chr = atac_seq[atac_seq["chrom"] == chr_str]
    atac_seq_chr = atac_seq_chr.sort_values(by=["start"], ascending =True)
    return atac_seq_chr

def generate(CELL_TYPE, SAVE_PATH):

    np.random.seed(0)

    if CELL_TYPE == "gm12878":
        chromosome_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] #gm12878
    elif CELL_TYPE == "k562":
        # chromosome_nums = [1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 19, 20, 21, 22] #k562
        chromosome_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] #k562
    elif CELL_TYPE == "hepg2":
        # chromosome_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22] #hepg2
        chromosome_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] #hepg2
    elif CELL_TYPE == "imr90":
        # chromosome_nums = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22] #imr90
        chromosome_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] #imr90
    else:
        #raise ValueError('This {} cell is not implemented'.format(CELL_TYPE))
        chromosome_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] #reed

    # chromosome_nums = [*range(1,23)]
    # Setting variable parameters
    # CELL_TYPES = ['gm12878']
    QUAN_NORM = True
    # ZIGZAG = True

    S_SEQUENCE = 401
    FLANK = int(np.floor(S_SEQUENCE / 2))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(DEVICE)

    def combine_input(path):
        COLUMNS = [
        'original_index',   # line number in the original atac file, 0-based
        'atacpeak',         # value of atacseq peak
        'chromStart',       # starting position of the feature in the chromosome 
        'chromEnd',         # ending position of the feature in the chromosome
        'center',           # position of the peak (max signal)
        ]

        COLUMNS.append('sequence')
        COLUMNS.append('dnase')
        COLUMNS.append('mappability')

        input_df = pd.read_parquet(f"{path}chr1.pq", columns=COLUMNS, engine='pyarrow').set_index('original_index')
        input_df['chrom'] = 'chr1'
        for chr in chromosome_nums:
            if chr == 1:
                continue
            print(f"{path}chr{chr}.pq")
            df = pd.read_parquet(f"{path}chr{chr}.pq", columns=COLUMNS, engine='pyarrow').set_index('original_index')
            df['chrom'] = f'chr{chr}'
            input_df = pd.concat([input_df, df], ignore_index=True)

        gc.collect()
        return input_df


    def prepare_input(chrom_num, atac_seq, dnase, map_seq):
        '''
        Preprocessing of input data and forming torch tensors that can be used directly by the transformers
        '''
        
        #normalize data:
        atac_seq.rename(columns={'chromStart': 'start', 'chromEnd': 'end'}, inplace=True)
        dnase.rename(columns={'chromStart': 'start', 'chromEnd': 'end'}, inplace=True)
        map_seq.rename(columns={'chromStart': 'start', 'chromEnd': 'end'}, inplace=True)
        coord_info = ['chrom', 'start', 'end', 'center']
            
        # data preprocess
        chrom_str = "chr{}".format(chrom_num)
        atac_seq_chr = data_augm.filter_sort_atac(atac_seq, chrom_str)
        map_seq_chr = data_augm.filter_sort_atac(map_seq, chrom_str)
        dnase_chr = data_augm.filter_sort_atac(dnase, chrom_str)

        input_df = atac_seq_chr.drop(columns=coord_info)
        input_df_dnase = dnase_chr.drop(columns=coord_info)
        input_df_map = map_seq_chr.drop(columns=coord_info)

        meta_df = atac_seq_chr[['chrom', 'center']]
        meta_df['chrom'] = meta_df['chrom'].apply(lambda x: int(x.split('chr')[-1]))

        # create dataset using atac_seq_chr and FLANK
        input_np = np.array(input_df, dtype='bool')
        input_np_dnase = np.array(input_df_dnase, dtype='float32')
        input_np_map = np.array(input_df_map, dtype='float32')
        meta_np = np.array(meta_df, dtype='float32')

        gc.collect()
        
        return input_np, meta_np, input_np_dnase, input_np_map


    def make_input_target_from_list(chrom_list, atac_seq, dnase, map_seq):
        
        X, meta, X_dnase, X_map = prepare_input(chrom_list[0], atac_seq, dnase, map_seq)
        indexing = np.arange(0,np.shape(X)[0]) + FLANK

        for train_chr in chrom_list[1:]:

            X_chr, meta_chr, X_dnase_chr, X_map_chr = prepare_input(train_chr, atac_seq, dnase, map_seq)

            indexing_chr = np.arange(0, np.shape(X_chr)[0]) + FLANK + np.shape(X)[0]
            indexing = np.concatenate((indexing, indexing_chr), axis=0)

            X = np.concatenate((X, X_chr), axis=0)
            X_dnase = np.concatenate((X_dnase, X_dnase_chr), axis=0)
            X_map = np.concatenate((X_map, X_map_chr), axis=0)
            meta = np.concatenate((meta, meta_chr), axis=0)

        return X, meta, indexing, X_dnase, X_map


    def save_data(chroms, feat_list, dnase_list, map_list, path):
        for atac_seq, dnase, map_seq in zip(feat_list, dnase_list, map_list):
            X, meta, indexing, X_dnase, X_map = make_input_target_from_list(chroms, atac_seq, dnase, map_seq)
        np.savez(path, dnase=X_dnase, sequence=X, meta=meta, indexing=indexing, mappability=X_map)

    ########################################
    ################# MAIN #################
    ########################################  

    feat_list, dnase_list, map_list = [], [], []
    # for CELL_TYPE in CELL_TYPES:
    if CELL_TYPE == "gm12878":
        # ATAC_PEAKS_PATH = "/mydata/mutiger/lin/mutiger-trepii/TREPII_torch/outputs/predict_gm12878_2024-04-16_11-46-17/"
        ATAC_PEAKS_PATH = "/cluster/work/boeva/tacisu/trepii/predict_gm12878_2024-06-26_09-51-17/"
    elif CELL_TYPE == "k562":
        # ATAC_PEAKS_PATH = "/mydata/mutiger/lin/mutiger-trepii/TREPII_torch/outputs/predict_k562_2024-03-18_15-39-51/"
        ATAC_PEAKS_PATH = "/cluster/work/boeva/tacisu/trepii/predict_k562_2024-06-26_10-16-27/"
    elif CELL_TYPE == "hepg2":
        # ATAC_PEAKS_PATH = "/mydata/mutiger/lin/mutiger-trepii/TREPII_torch/outputs/predict_hepg2_2024-04-19_09-00-07/"
        ATAC_PEAKS_PATH = "/cluster/work/boeva/tacisu/trepii/predict_hepg2_2024-06-26_11-30-29/"
    elif CELL_TYPE == "imr90":
        ATAC_PEAKS_PATH = "/cluster/work/boeva/tacisu/trepii/predict_imr90_2024-06-26_11-53-38/"
    else:
        ATAC_PEAKS_PATH = "/cluster/work/boeva/tacisu/trepii/predict_{}/".format(CELL_TYPE)
        # raise ValueError('This {} cell is not implemented'.format(CELL_TYPE))
    
    print(ATAC_PEAKS_PATH)

    atac_seq = combine_input(ATAC_PEAKS_PATH)
    atac_seq_dnase = atac_seq.copy(deep=True)
    atac_seq_map = atac_seq.copy(deep=True)

    n_feats = len(atac_seq.sequence.iloc[0])
    drop_cols = ['atacpeak', 'sequence', 'dnase', 'mappability']
    for i in range(n_feats):
        atac_seq[f'feat{i}'] = atac_seq['sequence'].apply(lambda x: x[i])
        
    n_feats = len(atac_seq_dnase.dnase.iloc[0])
    for i in range(n_feats):
        atac_seq_dnase[f'feat{i}'] = atac_seq_dnase['dnase'].apply(lambda x: x[i])

    n_feats = len(atac_seq_map.mappability.iloc[0])
    for i in range(n_feats):
        atac_seq_map[f'feat{i}'] = atac_seq_map['mappability'].apply(lambda x: x[i])

    atac_seq.drop(columns=drop_cols, inplace=True)
    atac_seq_dnase.drop(columns=drop_cols, inplace=True)
    atac_seq_map.drop(columns=drop_cols, inplace=True)

    feat_list.append(atac_seq)
    dnase_list.append(atac_seq_dnase)
    map_list.append(atac_seq_map)

    print("input data loaded!")

    gc.collect()

    test_path = f'{SAVE_PATH}/{CELL_TYPE.lower()}_test_meta_z_norm_pval_1.npz'    
    save_data(chromosome_nums, feat_list, dnase_list, map_list, test_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Stage-2 Data Generation")
    parser.add_argument('--cell_line', type=str, help='generate data for the specified cell line')
    parser.add_argument('--save_path', type=str, help='path where to store data')
    args = parser.parse_args()

    generate(args.cell_line, args.save_path)
