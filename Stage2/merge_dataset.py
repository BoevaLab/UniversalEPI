import sys, os 
if 'src' not in sys.path:
    sys.path.append('src')

import numpy as np
import pandas as pd
import argparse

# create blacklist masks for training

def generate_blacklist(save_path, cell_line, resolution):
    print(f"Generating blacklist for {cell_line}...")
    hic_path = os.path.join(os.getcwd(), 'data', 'hic', cell_line, 'raw_iced')
    chr_list, starts, ends = [], [], []
    for chr in range(1, 23):
        df = os.path.join(hic_path, f'chr{chr}_raw.bed')
        df['bin1'] = df.bin1//resolution
        df['bin2'] = df.bin2//resolution
        df['dist'] = df.bin2 - df.bin1
        x = df[df.dist==0].bin1.to_list()
        x1 = min(x)
        x2 = max(x)
        for i in range(x1, x2+1):
            if (i not in x) and (i-1 in x) and (i+1 in x):
                chr_list.append(f'chr{chr}')
                starts.append(i*resolution)
                ends.append(((i+1)*resolution)-1)
    
    blacklist = pd.DataFrame({
        'chr': chr_list,
        'start': starts,
        'end': ends
    })
    blacklist.to_csv(save_path, sep='\t', index=None)
    return blacklist


def merge_datasets(cell_lines, data_dir, phase, res, save_dir):

    target_list, dnase_list, seq_list, meta_list, idx_list, map_list, blacklist_list = [], [], [], [], [], [], []
    for cell_line in cell_lines:
        blacklist_path = os.path.path(os.getcwd(), 'data', 'blacklist', f'{cell_line}_blacklist.bed')
        if os.path.exists(blacklist_path):
            blacklist = pd.read_csv(blacklist_path, sep="\t", names=["chr", "start", "end"])
        else:
            blacklist = generate_blacklist(blacklist_path, cell_line, res)
        print(f"Blacklist for {cell_line} loaded successfully")

        data = np.load(os.path.join(data_dir, f'{cell_line}_{phase}.npz'))
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

        if phase == "val":
            test_chr = ["chr5", "chr21", "chr12", "chr13"] 
        else:
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

        target_list.append(data["target"])
        dnase_list.append(data["dnase"])
        seq_list.append(data["sequence"])
        meta_list.append(data["meta"])
        idx_list.append(data["indexing"])
        map_list.append(data["mappability"])
        blacklist_list.append(np.concatenate(mask_list))

    print('Merging datasets...')
    target = np.concatenate(target_list, axis=0)
    dnase = np.concatenate(dnase_list, axis=0)
    sequence = np.concatenate(seq_list, axis=0)
    meta = np.concatenate(meta_list, axis=0)
    indexing = np.concatenate(idx_list, axis=0)
    mappability = np.concatenate(map_list, axis=0)
    blacklist = np.concatenate(blacklist_list, axis=0)

    np.savez(os.path.join(save_dir, f"{phase}_dataset.npz"),
             target=target, 
             dnase=dnase, 
             sequence=sequence, 
             meta=meta, 
             indexing=indexing, 
             mappability=mappability, 
             blacklist=blacklist
            )
    print(f"Dataset for {phase} phase saved successfully")

def parse_args():
    parser = argparse.ArgumentParser(description="Create blacklist masks for training and merge datasets")
    parser.add_argument("--cell_lines", type=str, help="Cell lines", nargs='+', required=True)
    parser.add_argument("--data_dir", type=str, help="Path to the data directory", required=True)
    parser.add_argument("--save_dir", type=str, help="Path to save the merged dataset", default=None)
    parser.add_argument("--phase", type=str, default="train", help="Phase")
    parser.add_argument("--hic_res", type=int, default=5000, help="Resolution of Hi-C data")
    return parser.parse_args()


def main():
    args = parse_args()
    cell_lines = args.cell_lines
    phase = args.phase
    res = args.hic_res
    data_dir = args.data_dir
    save_dir = data_dir if args.save_dir is None else args.save_dir

    assert phase in ['train', 'val'], "Phase must be one of ['train', 'val']"
    merge_datasets(cell_lines, data_dir, phase, res, save_dir)

if __name__ == "__main__":
    main()