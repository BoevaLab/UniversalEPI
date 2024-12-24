import pickle as pkl
import pandas as pd
import numpy as np
import gc

import argparse


# python store_meta_data_hg38_dnase_seq.py --cell_line gm12878 --save_path /mydata/mutiger/shared/hg38_new_split/

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

def generate(cell_type, save_path, infer):

    np.random.seed(0)

    CV_FOLDS = {
            "fold_1": [5, 21, 12, 13],
            "fold_2": [19, 2, 6], 
            "fold_3": [3, 18, 16, 9, 20],
            "fold_4": [22, 8, 15, 7],
            "fold_5": [10, 4, 1],
            "fold_6": [11, 14]
    }

    chromosome_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    S_SEQUENCE = 401
    FLANK = int(np.floor(S_SEQUENCE / 2))

    def combine_input(path):
        COLUMNS = [
        'original_index',   # line number in the original atac file, 0-based
        'atacpeak',         # value of atacseq peak
        'chromStart',       # starting position of the feature in the chromosome 
        'chromEnd',         # ending position of the feature in the chromosome
        'center',           # position of the peak (max signal)
        'sequence',         # one-hot encoded sequence
        'dnase',            # ATAC signal
        'mappability',      # mappability signal
        ]

        input_df = pd.read_parquet(f"{path}chr1.pq", columns=COLUMNS, engine='pyarrow').set_index('original_index')
        input_df['chrom'] = 'chr1'
        for chr in chromosome_nums:
            if chr == 1:
                continue
            df = pd.read_parquet(f"{path}chr{chr}.pq", columns=COLUMNS, engine='pyarrow').set_index('original_index')
            df['chrom'] = f'chr{chr}'
            input_df = pd.concat([input_df, df], ignore_index=True)
        gc.collect()
        return input_df


    def add_data_to_dict(chrom_num, chrom_data_dict, cell_type, target_data_path):
        '''
        Create a mapping between chromosome number and its ground-truth matrix.
        '''
        if infer:
            target_path = "{}target_sequences_iced_rob_norm_var_smooth_chr{}.pkl".format(target_data_path, chrom_num)
        else:
            target_path = "{}target_sequences_dist_iced_robust_norm_with_neg_chr{}.pkl".format(target_data_path, chrom_num)
        
        
        try:
            with open(target_path, 'rb') as f:
                target_data = pkl.load(f)
        except:
            return chrom_data_dict
            
        chrom_data_dict[chrom_num] = target_data
        
        return chrom_data_dict

    def prepare_input(chrom_num, chrom_data_dict, atac_seq, dnase, map_seq):
        '''
        Preprocessing of input data and forming torch tensors that can be used directly by the transformers
        '''
        try:
            target_data = chrom_data_dict[chrom_num]
        except:
            return
        
        target_data = chrom_data_dict[chrom_num]
        
        #normalize data:
        atac_seq.rename(columns={'chromStart': 'start', 'chromEnd': 'end'}, inplace=True)
        dnase.rename(columns={'chromStart': 'start', 'chromEnd': 'end'}, inplace=True)
        map_seq.rename(columns={'chromStart': 'start', 'chromEnd': 'end'}, inplace=True)
        coord_info = ['chrom', 'start', 'end', 'center']
            
        # data preprocess
        chrom_str = "chr{}".format(chrom_num)
        atac_seq_chr = filter_sort_atac(atac_seq, chrom_str)
        map_seq_chr = filter_sort_atac(map_seq, chrom_str)
        dnase_chr = filter_sort_atac(dnase, chrom_str)

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
        
        Y = target_data.astype(np.float32)
        print(input_np.shape, Y.shape)
        
        gc.collect()
        
        return input_np, meta_np, Y, input_np_dnase, input_np_map


    def make_input_target_from_list(chrom_list, chrom_data_dict, atac_seq, dnase, map_seq):
        X, meta, Y, X_dnase, X_map = prepare_input(chrom_list[0], chrom_data_dict, atac_seq, dnase, map_seq)
        indexing = np.arange(0,np.shape(Y)[0]) + FLANK

        for train_chr in chrom_list[1:]:

            X_chr, meta_chr, Y_chr, X_dnase_chr, X_map_chr = prepare_input(train_chr, chrom_data_dict, atac_seq, dnase, map_seq)

            indexing_chr = np.arange(0, np.shape(Y_chr)[0]) + FLANK + np.shape(X)[0]
            indexing = np.concatenate((indexing, indexing_chr), axis=0)

            X = np.concatenate((X, X_chr), axis=0)
            X_dnase = np.concatenate((X_dnase, X_dnase_chr), axis=0)
            X_map = np.concatenate((X_map, X_map_chr), axis=0)
            meta = np.concatenate((meta, meta_chr), axis=0)
            Y = np.concatenate((Y, Y_chr), axis=0)
        return X, meta, Y, indexing, X_dnase, X_map


    def save_data(chroms, target_list, feat_list, dnase_list, map_list, path):
        for chrom_data_dict, atac_seq, dnase, map_seq in zip(target_list, feat_list, dnase_list, map_list):
            X, meta, Y, indexing, X_dnase, X_map = make_input_target_from_list(chroms, chrom_data_dict, atac_seq, dnase, map_seq)
        np.savez(path, target=Y, dnase=X_dnase, sequence=X, meta=meta, indexing=indexing, mappability=X_map)
    
    ########################################
    ################# MAIN #################
    ########################################  

    target_list, feat_list, dnase_list, map_list = [], [], [], []
    
    ATAC_PEAKS_PATH = "/cluster/work/boeva/tacisu/trepii/predict_{}_swap_cell_neg/".format(cell_type.lower())

    TARGET_DATA_PATH = "/cluster/work/boeva/tacisu/trepi/new_data/{}/target_data_zigzag/".format(cell_type)  #hic_z_norm hg38_targets

    print(TARGET_DATA_PATH)

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

    chrom_data_dict = {}
    for chr_num in chromosome_nums:
        chrom_data_dict = add_data_to_dict(chr_num, chrom_data_dict, cell_type, TARGET_DATA_PATH)
    target_list.append(chrom_data_dict)
    gc.collect()


    for cv_fold in range(1, 2):
        print("Cross validation fold_{}!".format(cv_fold))

        val_chroms_fold = "fold_{}".format(cv_fold)
        VAL_CHROMS = CV_FOLDS[val_chroms_fold]
        
        holdout_chroms_fold = "fold_{}".format((cv_fold%6)+1)
        HOLDOUT_CHROMS = CV_FOLDS[holdout_chroms_fold]
        
        TRAIN_CHROMS = [chr_n for chr_n in chromosome_nums if (chr_n not in VAL_CHROMS) and (chr_n not in HOLDOUT_CHROMS)]
        
        print("Training chromosomes: ", TRAIN_CHROMS)
        print("VAL_CHROMS:", VAL_CHROMS, "HOLDOUT_CHROMS:", HOLDOUT_CHROMS)

        train_path = f'{save_path}/{cell_type.lower()}_train_meta_z_norm_pval_{cv_fold}.npz'
        val_path = f'{save_path}/{cell_type.lower()}_val_meta_z_norm_pval_{cv_fold}.npz'
        test_path = f'{save_path}/{cell_type.lower()}_test_meta_z_norm_pval_{cv_fold}.npz'
        
        if not infer:
            save_data(TRAIN_CHROMS, target_list, feat_list, dnase_list, map_list, train_path)
            save_data(VAL_CHROMS, target_list, feat_list, dnase_list, map_list, val_path)
        else:
            save_data(HOLDOUT_CHROMS, target_list, feat_list, dnase_list, map_list, test_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Stage-2 Data Generation")
    parser.add_argument('--cell_line', type=str, help='generate data for the specified cell line')
    parser.add_argument('--save_path', type=str, help='path where to store data')
    parser.add_argument('--infer', action='store_true', help='generate only test data')
    args = parser.parse_args()

    generate(args.cell_line, args.save_path, args.infer)
