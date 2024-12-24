import sys, os 
if 'src' not in sys.path:
    sys.path.append('src')

import numpy as np
import pandas as pd

# create blacklist masks for training

cell_lines = ["hepg2", "imr90"]
phases = ["train", "val"]

for cell_line in cell_lines:
    for phase in phases:
        blacklist = pd.read_csv(f'/cluster/work/boeva/tacisu/trepi/new_data/blacklist/{cell_line}_blacklist.bed', sep="\t", names=["chr", "start", "end"])

        data = np.load(f'/cluster/work/boeva/tacisu/trepi/{cell_line}_{phase}_meta_z_norm_pval_swap_1.npz')
        target = data["target"][:,1:201]
        meta = data["meta"]
        indexing = data["indexing"]
        FLANK = 200
        pos = np.zeros((indexing.shape[0], 401))
        chr_data = np.zeros((indexing.shape[0], 401))

        for i in range(indexing.shape[0]):
            data_idx = indexing[i]
            m = meta[data_idx-FLANK:data_idx+FLANK+1,:]
            pos[i,:] = m[:,-1]
            chr_data[i,:] = m[:,-2]

        ind_a = np.arange(200,99,-1).repeat(2)
        ind_b = np.arange(200,301).repeat(2)
        ind_a = ind_a[1:]
        ind_b = ind_b[:-1]

        res = 5000
        rel_pos = np.abs((pos[:,ind_a]//res)-(pos[:,ind_b]//res))

        pos1 = (pos[:,ind_a]//res)
        pos2 = (pos[:,ind_b]//res)
        chr_data = chr_data[:,ind_b]
        pos1 = pos1[:,1:]
        pos2 = pos2[:,1:]
        chr_data = chr_data[:,1:]
        rel_pos = rel_pos[:,1:]

        result_df = pd.DataFrame({
            'chrom': chr_data.flatten(),
            'bin1': pos1.flatten(),
            'bin2': pos2.flatten(),
            'target': target.flatten(),
        })

        
        if phase == "test":
            test_chr = ["chr19", "chr2", "chr6"]
        elif phase == "val":
            test_chr = ["chr5", "chr21", "chr12", "chr13"] 
        elif phase == "train":
            test_chr = ["chr1", "chr3", "chr4", "chr7", "chr8", "chr9", "chr10", "chr11", "chr14",
                        "chr15", "chr16", "chr17", "chr18", "chr22", "chr20"] 

        blacklist = blacklist[blacklist.chr.isin(test_chr)]
        blacklist['start'] = blacklist['start'].apply(lambda x: int(x))
        drop_info = {}
        for chr in test_chr:
            blacklist_chr = blacklist[blacklist.chr == chr]
            chr = int(chr.split("chr")[-1])
            blacklist_chr['bin1'] = blacklist_chr.start//res
            drop_list = blacklist_chr.bin1.to_list()
            drop_info[chr] = drop_list

        mask_list = []
        for chr, drop_list in drop_info.items():
            df_filter = result_df[result_df.chrom==chr]
            mask1 = np.array(df_filter.bin1.isin(drop_list))
            mask2 = np.array(df_filter.bin2.isin(drop_list))
            mask = (mask1 | mask2)
            mask = np.reshape(mask, (-1,200))
            mask_list.append(mask)

        print(np.sum(1-np.concatenate(mask_list)))

        val_path = f'/cluster/work/boeva/tacisu/trepi/{cell_line}_{phase}_meta_z_norm_pval_1.npz'
        np.savez(val_path, target=data["target"], dnase=data["dnase"], 
                sequence=data["sequence"], meta=data["meta"], 
                indexing=data["indexing"], mappability=data["mappability"],
                blacklist=np.concatenate(mask_list))
        
data1 = np.load("/cluster/work/boeva/tacisu/trepi/imr90_train_meta_z_norm_pval_swap_1.npz")
data2 = np.load("/cluster/work/boeva/tacisu/trepi/hepg2_train_meta_z_norm_pval_swap_1.npz")

val_path = "/cluster/work/boeva/tacisu/trepi/train_meta_z_norm_pval_swap.npz"
np.savez(val_path, target=np.concatenate((data1["target"], data2["target"]), axis=0), dnase=np.concatenate((data1["dnase"], data2["dnase"]), axis=0), 
         sequence=np.concatenate((data1["sequence"], data2["sequence"]), axis=0), meta=np.concatenate((data1["meta"], data2["meta"]), axis=0), 
         indexing=np.concatenate((data1["indexing"], data2["indexing"]+data1["sequence"].shape[0]), axis=0),
         mappability=np.concatenate((data1["mappability"], data2["mappability"]), axis=0),
         blacklist=np.concatenate((data1["blacklist"], data2["blacklist"]), axis=0))

data1 = np.load("/cluster/work/boeva/tacisu/trepi/imr90_val_meta_z_norm_pval_swap_1.npz")
data2 = np.load("/cluster/work/boeva/tacisu/trepi/hepg2_val_meta_z_norm_pval_swap_1.npz")

val_path = "/cluster/work/boeva/tacisu/trepi/val_meta_z_norm_pval_swap.npz"
np.savez(val_path, target=np.concatenate((data1["target"], data2["target"]), axis=0), dnase=np.concatenate((data1["dnase"], data2["dnase"]), axis=0), 
         sequence=np.concatenate((data1["sequence"], data2["sequence"]), axis=0), meta=np.concatenate((data1["meta"], data2["meta"]), axis=0), 
         indexing=np.concatenate((data1["indexing"], data2["indexing"]+data1["sequence"].shape[0]), axis=0),
         mappability=np.concatenate((data1["mappability"], data2["mappability"]), axis=0),
         blacklist=np.concatenate((data1["blacklist"], data2["blacklist"]), axis=0))
