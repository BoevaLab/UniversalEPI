import os
import sys

if "src" not in sys.path:
    sys.path.append("src")

import argparse

import numpy as np
import torch
from create_dataset import generate as generate_datasets
from dataset import NPZDatasetRaw
from deepc.trepii_model import DeepC
from models.transformer_encoder_model import Transformer_Encoder
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
from utilities import getConfig


# python eval.py --config_dir configs/configs.yaml
# ----------------------------------------------------------------------------
def eval(opt, cell_line):

    if torch.cuda.is_available():
        use_gpu = True
        device = torch.device("cuda")
    else:
        use_gpu = False
        device = torch.device("cpu")

    model = Transformer_Encoder(
        input_dim=opt.n_feat,
        batch_size=opt.batch_size,
        input_feat_dims=opt.input_feat_dims,
        relative_positions=opt.RELATIVE_POSITIONS,
        meta_flg=opt.META,
        peu_flg=opt.PEU,
        stg_flg=opt.STG,
        map_flg=opt.MAP,
        atac_flg=opt.ATAC,
        var_flg=opt.var_flg,
        max_len=opt.max_len,
        pe_res=opt.pe_res,
        seq_len=opt.seq_len,
        binning=opt.binning,
        num_heads=opt.num_heads,
        num_layers=opt.num_layers,
        embed_dim=opt.embed_dim,
        hidden_dim=opt.hidden_dim,
        dropout=opt.dropout,
        get_attn=opt.ATTENTION,
        device=device,
    )

    artifact_dir = f"{opt.logs_dir}/{opt.run_name}"

    model_path = os.path.join(artifact_dir, "checkpoint.pth")
    weights = torch.load(model_path, map_location="cpu")
    model.load_state_dict(weights, strict=False)
    model.eval()

    featNet = DeepC(opt.ksizes, opt.channels, opt.poolings)
    weights = torch.load(opt.stage_1_model, map_location="cpu")
    featNet.load_state_dict(weights)
    featNet.eval()

    if use_gpu:
        model.to(device)
        featNet.to(device)

    if opt.test_dir is None:
        atac_path = os.path.join(opt.atac_path, f"predict_{cell_line}")
        mode = "test"
        dataset_test = NPZDatasetRaw(
            data=generate_datasets(atac_path, opt.hic_path, opt.data_save_dir, mode, opt.seq_len, opt.chroms)
        )
    else:
        dataset_test = NPZDatasetRaw(data_dir=opt.test_dir)

    dataloader_test = DataLoader(
        dataset_test, shuffle=False, batch_size=1, drop_last=False, num_workers=opt.num_workers
    )

    len_dataset = len(dataset_test)
    print("Length of dataset: ", len_dataset)

    ind_a = np.arange(200, 99, -1).repeat(2)
    ind_b = np.arange(200, 301).repeat(2)
    ind_a = ind_a[1:]
    ind_b = ind_b[:-1]

    predictions = []
    targets = []
    variance = []
    chr = []
    pos1 = []
    pos2 = []

    mse = []

    with torch.no_grad():
        for [dnase_i_val, seq_i_val, target_i_val, meta_i_val, map_i_val, _] in tqdm(dataloader_test):
            if use_gpu:
                dnase_i_val = dnase_i_val.to(device)
                seq_i_val = seq_i_val.to(device)
                target_i_val = target_i_val.to(device)
                meta_i_val = meta_i_val.to(device)
                map_i_val = map_i_val.to(device)

            dnase_i_flatten_val = torch.flatten(dnase_i_val, end_dim=1).unsqueeze(1)
            seq_i_val = torch.flatten(seq_i_val, end_dim=1)
            data_i_val = featNet(torch.cat((dnase_i_flatten_val, seq_i_val), dim=1), feat=True)
            output, _, _ = model(data_i_val, map_i_val, meta_i_val, dnase_i_val)

            if opt.var_flg:
                out_sigma = torch.exp(output[:, 1])
                output = output[:, 0]

            if opt.log_scale:
                target_i_val = torch.log(target_i_val + 1)

            mse.extend(torch.mean(torch.square(output - target_i_val[:, 1:201]), axis=-1).cpu().detach().numpy())

            variance_list = list(out_sigma.cpu().detach().numpy().flatten())
            output_list = list(output.cpu().detach().numpy().flatten())
            target_list = list(target_i_val[:, 1:201].cpu().numpy().flatten())
            pos = meta_i_val[:, :, -1].detach().cpu().numpy()
            pos1_i = pos[:, ind_a]
            pos2_i = pos[:, ind_b]
            pos1_list = list(pos1_i[:, 1:].flatten())
            pos2_list = list(pos2_i[:, 1:].flatten())

            chr_i = meta_i_val[:, :, -2].detach().cpu().numpy()
            chr_i = chr_i[:, ind_b]
            chr_list = list(chr_i[:, 1:].flatten())

            predictions.extend(output_list)
            targets.extend(target_list)
            variance.extend(variance_list)
            chr.extend(chr_list)
            pos1.extend(pos1_list)
            pos2.extend(pos2_list)

        val_corr = np.corrcoef(np.array(targets).flatten(), np.array(predictions).flatten())[0, 1]
        print("PEARSON CORRELATION: ", val_corr)

        val_rank_corr, p_val = spearmanr(np.array(targets).flatten(), np.array(predictions).flatten())
        print("SPEARMAN CORRELATION + PVAL: ", val_rank_corr, p_val)

        val_rank_corr, p_val = kendalltau(np.array(targets).flatten(), np.array(predictions).flatten())
        print("Kendall CORRELATION + PVAL: ", val_rank_corr, p_val)

        mse_val = np.mean(np.array(mse))
        print("Mean Squared Error: ", mse_val)

    if opt.save_res:
        save_dir = os.path.join(opt.res_dir, cell_line, opt.run_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez(
            os.path.join(save_dir, "results.npz"),
            targets=np.array(targets),
            predictions=np.array(predictions),
            variance=np.array(variance),
            chr=np.array(chr),
            pos1=np.array(pos1),
            pos2=np.array(pos2),
        )

    print("eval done")


if __name__ == "__main__":
    os.environ["KMP_WARNINGS"] = "0"

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--config_dir", type=str, help="Root directory for training configs.", required=True)

    args = parser.parse_args()
    config = getConfig(args.config_dir)
    cell_line = (config.test_dir.split("/")[-1]).split("_")[0]

    eval(config, cell_line)
