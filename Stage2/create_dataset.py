import argparse
import gc
import os
import pickle as pkl
import warnings

import numpy as np
import pandas as pd

"""
Create an input dataset for training and validation.
    ```
    python ./Stage2/create_dataset.py -g ./data/stage1_outputs/predict_gm12878 -s ./data/processed_data/ --hic_data_dir ./data/hic/ --mode train
    python ./Stage2/create_dataset.py -g ./data/stage1_outputs/predict_k562 -s ./data/processed_data/ --hic_data_dir ./data/hic/ --mode train
    python ./Stage2/create_dataset.py -g ./data/stage1_outputs/predict_gm12878 -s ./data/processed_data/ --hic_data_dir ./data/hic/ --mode val
    python ./Stage2/create_dataset.py -g ./data/stage1_outputs/predict_k562 -s ./data/processed_data/ --hic_data_dir ./data/hic/ --mode val
    ```
    This creates `gm12878_train.npz`, `k562_train.npz`, `gm12878_val.npz`, and `k562_train.npz` in `./data/processed_data`.
"""

chromosome_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


def filter_sort_atac(atac_seq, chr_str):
    """
    takes pandas.DataFrame of all atac-seq peaks and filters for one
    chromosome as well as sorts them ascending by start nucleotide
    """
    try:
        atac_seq_chr = atac_seq[atac_seq["chrom"] == chr_str].drop(["sth2", "sth3"], axis=1)
    except:
        atac_seq_chr = atac_seq[atac_seq["chrom"] == chr_str]
    atac_seq_chr = atac_seq_chr.sort_values(by=["start"], ascending=True)
    return atac_seq_chr


def combine_input(path):
    COLUMNS = [
        "original_index",  # line number in the original atac file, 0-based
        "atacpeak",  # value of atacseq peak
        "chromStart",  # starting position of the feature in the chromosome
        "chromEnd",  # ending position of the feature in the chromosome
        "center",  # position of the peak (max signal)
        "sequence",  # one-hot encoded sequence
        "dnase",  # ATAC signal
        "mappability",  # mappability signal
    ]
    df_chr1_path = os.path.join(path, "chr1.pq")
    input_df = pd.read_parquet(df_chr1_path, columns=COLUMNS, engine="pyarrow").set_index("original_index")
    input_df["chrom"] = "chr1"
    for chr in chromosome_nums:
        if chr == 1:
            continue
        df_path = os.path.join(path, f"chr{chr}.pq")
        df = pd.read_parquet(df_path, columns=COLUMNS, engine="pyarrow").set_index("original_index")
        df["chrom"] = f"chr{chr}"
        input_df = pd.concat([input_df, df], ignore_index=True)
    gc.collect()
    return input_df


def add_data_to_dict(chrom_num, chrom_data_dict, target_data_path, cell_line):
    """
    Create a mapping between chromosome number and its ground-truth matrix.
    """

    target_path = os.path.join(target_data_path, cell_line, f"chr{chrom_num}_target.pkl")

    try:
        with open(target_path, "rb") as f:
            target_data = pkl.load(f)
    except:
        return chrom_data_dict

    chrom_data_dict[chrom_num] = target_data

    return chrom_data_dict


def prepare_input(chrom_num, chrom_data_dict, atac_seq, dnase, map_seq):
    """
    Preprocessing of input data and forming torch tensors that can be used directly by the transformers
    """
    if chrom_data_dict is not None:
        try:
            target_data = chrom_data_dict[chrom_num]
        except:
            raise ValueError(f"Chromosome {chrom_num} not found in target data")
        Y = target_data.astype(np.float32)
    else:
        Y = None

    # normalize data:
    atac_seq.rename(columns={"chromStart": "start", "chromEnd": "end"}, inplace=True)
    dnase.rename(columns={"chromStart": "start", "chromEnd": "end"}, inplace=True)
    map_seq.rename(columns={"chromStart": "start", "chromEnd": "end"}, inplace=True)
    coord_info = ["chrom", "start", "end", "center"]

    # data preprocess
    chrom_str = "chr{}".format(chrom_num)
    atac_seq_chr = filter_sort_atac(atac_seq, chrom_str)
    map_seq_chr = filter_sort_atac(map_seq, chrom_str)
    dnase_chr = filter_sort_atac(dnase, chrom_str)

    input_df = atac_seq_chr.drop(columns=coord_info)
    input_df_dnase = dnase_chr.drop(columns=coord_info)
    input_df_map = map_seq_chr.drop(columns=coord_info)

    meta_df = atac_seq_chr[["chrom", "center"]]
    meta_df["chrom"] = meta_df["chrom"].apply(lambda x: int(x.split("chr")[-1]))

    # create dataset using atac_seq_chr and FLANK
    input_np = np.array(input_df, dtype="bool")
    input_np_dnase = np.array(input_df_dnase, dtype="float32")
    input_np_map = np.array(input_df_map, dtype="float32")
    meta_np = np.array(meta_df, dtype="float32")

    gc.collect()

    return input_np, meta_np, Y, input_np_dnase, input_np_map


def make_input_target_from_list(chrom_list, chrom_data_dict, atac_seq, dnase, map_seq, flank):
    X, meta, Y, X_dnase, X_map = prepare_input(chrom_list[0], chrom_data_dict, atac_seq, dnase, map_seq)
    indexing = np.arange(0, np.shape(Y)[0]) + flank

    for train_chr in chrom_list[1:]:

        X_chr, meta_chr, Y_chr, X_dnase_chr, X_map_chr = prepare_input(
            train_chr, chrom_data_dict, atac_seq, dnase, map_seq
        )

        indexing_chr = np.arange(0, np.shape(Y_chr)[0]) + flank + np.shape(X)[0]
        indexing = np.concatenate((indexing, indexing_chr), axis=0)

        X = np.concatenate((X, X_chr), axis=0)
        X_dnase = np.concatenate((X_dnase, X_dnase_chr), axis=0)
        X_map = np.concatenate((X_map, X_map_chr), axis=0)
        meta = np.concatenate((meta, meta_chr), axis=0)
        if chrom_data_dict is not None:
            Y = np.concatenate((Y, Y_chr), axis=0)
    if Y is None:
        return X, meta, indexing, X_dnase, X_map
    return X, meta, Y, indexing, X_dnase, X_map


def save_data(chroms, target_list, feat_list, dnase_list, map_list, path, flank):
    data = {}
    if len(target_list) == 0:
        for atac_seq, dnase, map_seq in zip(feat_list, dnase_list, map_list):
            X, meta, indexing, X_dnase, X_map = make_input_target_from_list(
                chroms, None, atac_seq, dnase, map_seq, flank
            )
        data["dnase"] = X_dnase
        data["sequence"] = X
        data["meta"] = meta
        data["indexing"] = indexing
        data["mappability"] = X_map
        if path:
            np.savez(path, dnase=X_dnase, sequence=X, meta=meta, indexing=indexing, mappability=X_map)
    else:
        for chrom_data_dict, atac_seq, dnase, map_seq in zip(target_list, feat_list, dnase_list, map_list):
            X, meta, Y, indexing, X_dnase, X_map = make_input_target_from_list(
                chroms, chrom_data_dict, atac_seq, dnase, map_seq, flank
            )
        data["target"] = Y
        data["dnase"] = X_dnase
        data["sequence"] = X
        data["meta"] = meta
        data["indexing"] = indexing
        data["mappability"] = X_map
        if path:
            np.savez(path, target=Y, dnase=X_dnase, sequence=X, meta=meta, indexing=indexing, mappability=X_map)
    return data


def generate(atac_path, hic_path, save_dir, mode, seq_len=401, chrs=None):

    np.random.seed(0)

    flank = int(np.floor(seq_len / 2))
    cell_line = (atac_path.rstrip("/").split("/")[-1]).split("_")[1]

    target_list, feat_list, dnase_list, map_list = [], [], [], []

    atac_seq = combine_input(atac_path)
    atac_seq_dnase = atac_seq.copy(deep=True)
    atac_seq_map = atac_seq.copy(deep=True)

    n_feats = len(atac_seq.sequence.iloc[0])
    drop_cols = ["atacpeak", "sequence", "dnase", "mappability"]
    for i in range(n_feats):
        atac_seq[f"feat{i}"] = atac_seq["sequence"].apply(lambda x: x[i])

    n_feats = len(atac_seq_dnase.dnase.iloc[0])
    for i in range(n_feats):
        atac_seq_dnase[f"feat{i}"] = atac_seq_dnase["dnase"].apply(lambda x: x[i])

    n_feats = len(atac_seq_map.mappability.iloc[0])
    for i in range(n_feats):
        atac_seq_map[f"feat{i}"] = atac_seq_map["mappability"].apply(lambda x: x[i])

    atac_seq.drop(columns=drop_cols, inplace=True)
    atac_seq_dnase.drop(columns=drop_cols, inplace=True)
    atac_seq_map.drop(columns=drop_cols, inplace=True)

    feat_list.append(atac_seq)
    dnase_list.append(atac_seq_dnase)
    map_list.append(atac_seq_map)

    if hic_path is not None:
        chrom_data_dict = {}
        for chr_num in chromosome_nums:
            chrom_data_dict = add_data_to_dict(chr_num, chrom_data_dict, hic_path, cell_line)
        target_list.append(chrom_data_dict)
    gc.collect()

    val_chroms = [5, 21, 12, 13]
    test_chroms = [19, 6, 2]
    train_chroms = [chr for chr in chromosome_nums if chr not in val_chroms + test_chroms]
    if mode == "test":
        chroms = test_chroms
    elif mode == "val":
        chroms = val_chroms
    else:
        chroms = train_chroms

    if save_dir:
        save_path = os.path.join(save_dir, f"{cell_line}_{mode}.npz")
    else:
        save_path = None

    if hic_path is None:
        if chrs is not None:
            chroms = chrs
        else:
            chroms = chromosome_nums
        if save_dir:
            save_path = os.path.join(save_dir, f"{cell_line}_input.npz")
        else:
            save_path = None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    data = save_data(chroms, target_list, feat_list, dnase_list, map_list, save_path, flank)

    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Stage2 Data Generation")
    parser.add_argument("-g", "--genomic_data_path", type=str, help="path to the genomic data", required=True)
    parser.add_argument("-s", "--save_dir", type=str, help="path where to store data", required=True)
    parser.add_argument("-m", "--mode", type=str, help="mode of data generation", default="test")
    parser.add_argument("--hic_data_dir", type=str, help="path to the hic data", default=None)
    parser.add_argument("--seq_len", type=int, help="sequence length", default=401)
    parser.add_argument("--chroms", nargs="+", type=int, help="chromosomes to generate data for", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    atac_path = args.genomic_data_path
    hic_path = args.hic_data_path
    save_dir = args.save_dir
    mode = args.mode
    seq_len = args.seq_len
    chroms = args.chroms

    assert mode in ["train", "val", "test"], "Mode must be one of ['train', 'val', 'test']"
    if mode != "test":
        assert hic_path is not None, "hic_data_path must be provided if mode is not 'test'"

    generate(atac_path, hic_path, save_dir, mode, seq_len, chroms)


if __name__ == "__main__":
    main()
