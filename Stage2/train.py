import os
import sys

if "src" not in sys.path:
    sys.path.append("src")

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from create_dataset import generate as generate_datasets
from dataset import NPZDatasetRaw
from deepc.trepii_model import DeepC
from merge_dataset import merge_datasets
from models.transformer_encoder_model import Transformer_Encoder
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader
from utilities import IterationCounter, Visualizer, getConfig


# python train.py --config_dir configs/configs.yaml
# ----------------------------------------------------------------------------
def train(opt):

    # -----------------------Initialize---------------------------------------
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    print(f"Random Seed: {opt.seed}")
    print(f"{opt.run_name}")

    if torch.cuda.is_available():
        use_gpu = True
        device = torch.device("cuda:0")
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

    featNet = DeepC(opt.ksizes, opt.channels, opt.poolings)
    weights = torch.load(opt.stage_1_model, map_location="cpu")
    featNet.load_state_dict(weights)
    featNet.eval()

    print("#params of the HiC model")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("#params of the featNet")
    print(sum(p.numel() for p in featNet.parameters() if p.requires_grad))

    if use_gpu:
        model.to(device)
        featNet.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    # -----------------------Loading Datasets----------------------------------
    if opt.train_dir is None:
        data_dict = {}
        mode = "train"
        for cell_line in opt.cell_lines_train:
            atac_path_cell = os.path.join(opt.atac_path, f"predict_{cell_line}")
            data_dict[f"{cell_line}_{mode}"] = generate_datasets(
                atac_path_cell, opt.hic_path, opt.data_save_dir, mode, opt.seq_len, opt.chroms
            )
        data_merged = merge_datasets(
            opt.cell_lines_train, mode, opt.hic_res, data_dict=data_dict, save_dir=opt.data_save_dir
        )
        dataset_train = NPZDatasetRaw(data=data_merged)
    else:
        dataset_train = NPZDatasetRaw(data_dir=opt.train_dir)

    if opt.val_dir is None:
        data_dict = {}
        mode = "val"
        for cell_line in opt.cell_lines_val:
            atac_path_cell = os.path.join(opt.atac_path, f"predict_{cell_line}")
            data_dict[f"{cell_line}_{mode}"] = generate_datasets(
                atac_path_cell, opt.hic_path, opt.data_save_dir, mode, opt.seq_len, opt.chroms
            )
        data_merged = merge_datasets(
            opt.cell_lines_val, mode, opt.hic_res, data_dict=data_dict, save_dir=opt.data_save_dir
        )
        dataset_val = NPZDatasetRaw(data=data_merged)
    else:
        dataset_val = NPZDatasetRaw(data_dir=opt.val_dir)

    batch_size = opt.batch_size

    print("Loading train data")
    dataloader_train = DataLoader(
        dataset_train, shuffle=True, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers
    )
    print("Loading val data")
    dataloader_val = DataLoader(
        dataset_val, shuffle=False, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers
    )

    len_dataset = len(dataset_train)
    print("Length of dataset: ", len_dataset)
    print("Length of val dataset: ", len(dataset_val))

    # create tool for counting iterations
    visualizer = Visualizer(opt)
    iter_counter = IterationCounter(opt, len_dataset)
    iter_counter.set_batchsize(batch_size)

    best_corr = -1

    # -----------------------Training loop----------------------------------
    for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        print("Starting epoch ", epoch)

        # Loop over training data
        for i, [dnase_i, seq_i, target_i, meta_i, map_i, mask_i] in enumerate(
            dataloader_train, start=iter_counter.epoch_iter
        ):
            iter_counter.record_one_iteration()

            if use_gpu:
                dnase_i = dnase_i.to(device)
                seq_i = seq_i.to(device)
                target_i = target_i.to(device)
                meta_i = meta_i.to(device)
                map_i = map_i.to(device)
                mask_i = mask_i.to(device)

            if opt.log_scale:
                target_i = torch.log(target_i + 1)

            dnase_i_flatten = torch.flatten(dnase_i, end_dim=1).unsqueeze(1)
            seq_i = torch.flatten(seq_i, end_dim=1)
            data_i = featNet(torch.cat((dnase_i_flatten, seq_i), dim=1), feat=True)

            optimizer.zero_grad()
            output, _, loss_reg = model(data_i, map_i, meta_i, dnase_i)

            if opt.var_flg:
                out_sigma = torch.exp(output[:, :, 1])
                output = output[:, :, 0]

                out_sigma = out_sigma[mask_i < 0.5]
                out_mean = output[mask_i < 0.5]
                target_mask = target_i[:, 1:201]
                target_mask = target_mask[mask_i < 0.5]

                beta = opt.beta
                exponent = 0.5 * torch.log(out_sigma) + (out_mean - target_mask) ** 2 / (2 * out_sigma)
                exponent = exponent * out_sigma.detach() ** beta

            if opt.var_flg:
                loss_main = torch.mean(exponent)
            else:
                loss_main = criterion(output * (1 - mask_i), target_i[:, 1:201] * (1 - mask_i))

            loss = loss_main + opt.stg_reg * loss_reg

            loss.backward()
            optimizer.step()

            output_list = output[mask_i < 0.5]
            target_list = target_i[:, 1:201]
            target_list = target_list[mask_i < 0.5]

            output_list = list(output_list.cpu().detach().numpy().flatten())
            target_list = list(target_list.cpu().numpy().flatten())
            batch_corr, _ = spearmanr(np.array(output_list).flatten(), np.array(target_list).flatten())
            batch_corr_pearson = np.corrcoef(np.array(output_list).flatten(), np.array(target_list).flatten())[0, 1]

            if iter_counter.needs_printing():
                losses = {
                    "loss_all": loss.detach().item(),
                    "loss": loss_main.detach().item(),
                    "spearman": batch_corr,
                    "pearson": batch_corr_pearson,
                }

                if opt.STG:
                    losses.update({"loss_reg": loss_reg.detach().item()})

                visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        scheduler.step()

        # Loop over validation data
        with torch.no_grad():

            total_corr = 0.0
            pearson = 0.0
            kendall = 0.0
            total_val_loss = 0.0

            for i, [dnase_i_val, seq_i_val, target_i_val, meta_i_val, map_i_val, mask_i_val] in enumerate(
                dataloader_val
            ):

                if use_gpu:
                    dnase_i_val = dnase_i_val.to(device)
                    seq_i_val = seq_i_val.to(device)
                    target_i_val = target_i_val.to(device)
                    meta_i_val = meta_i_val.to(device)
                    map_i_val = map_i_val.to(device)
                    mask_i_val = mask_i_val.to(device)

                if opt.log_scale:
                    target_i_val = torch.log(target_i_val + 1)

                dnase_i_flatten_val = torch.flatten(dnase_i_val, end_dim=1).unsqueeze(1)
                seq_i_val = torch.flatten(seq_i_val, end_dim=1)
                data_i_val = featNet(torch.cat((dnase_i_flatten_val, seq_i_val), dim=1), feat=True)
                output, _, _ = model(data_i_val, map_i_val, meta_i_val, dnase_i_val)

                if opt.var_flg:
                    out_sigma = torch.exp(output[:, :, 1])
                    output = output[:, :, 0]

                val_loss = criterion(output * (1 - mask_i_val), target_i_val[:, 1:201] * (1 - mask_i_val))
                total_val_loss += val_loss

                output_list = output[mask_i_val < 0.5]
                target_list = target_i_val[:, 1:201]
                target_list = target_list[mask_i_val < 0.5]

                output_list = list(output_list.cpu().detach().numpy().flatten())
                target_list = list(target_list.cpu().numpy().flatten())

                batch_corr, _ = spearmanr(np.array(output_list).flatten(), np.array(target_list).flatten())
                total_corr += batch_corr

                kendall_corr, _ = kendalltau(np.array(output_list).flatten(), np.array(target_list).flatten())
                kendall += kendall_corr

                pearson_corr = np.corrcoef(np.array(output_list).flatten(), np.array(target_list).flatten())[0, 1]
                pearson += pearson_corr

            val_corr = total_corr / (i + 1)

            losses = {
                "spearman_corr_val": val_corr,
                "kendall_corr_val": kendall / (i + 1),
                "pearson_corr_val": pearson / (i + 1),
                "loss_val": total_val_loss / (i + 1),
            }

            if val_corr > best_corr:
                best_corr = val_corr
                save_path = os.path.join(opt.logs_dir, opt.run_name, "checkpoint.pth")
                torch.save(model.state_dict(), save_path)
                print("best model: epoch {}, corr {}".format(epoch, val_corr))

            visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
            visualizer.display_current_results(
                output[:4, :], target_i_val[:4, 1:201], meta_i_val[:4, 1:201, 1:], iter_counter.total_steps_so_far
            )


if __name__ == "__main__":
    os.environ["KMP_WARNINGS"] = "0"

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config_dir", type=str, help="Root directory for training configs.")

    args = parser.parse_args()
    config = getConfig(args.config_dir)

    train(config)
