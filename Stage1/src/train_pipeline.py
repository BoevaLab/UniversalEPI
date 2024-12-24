from typing import Optional
import argparse
import os
import hydra
from omegaconf import DictConfig

from src import utils
from src.datamodules.encode_datamodule import MultiCellModule

from src.models.trepii_model import DeepC

import numpy as np
import torch


def get_logger():
    log = utils.get_logger(__name__)
    return log

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--save_path', type=str, help='path to save the model', required=True)
    return parser.parse_args()

def run(config: DictConfig) -> Optional[float]:
    args = parse_args()
    save_path = args.save_path

    log = get_logger()

    # Init model
    model = DeepC()
    criterion = torch.nn.MSELoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001, eps=1e-8, betas=[0.9, 0.999])

    # Init datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: MultiCellModule = hydra.utils.instantiate(
        config.datamodule, _recursive_=False
    )

    # The following steps must be run for each of the training phases. Datamodule
    # supports:
    # - fit
    # - validate
    # - test
    # - predict

    # Setup training stage of data module. This will load all datasets associated
    # with the training set.
    # IMPORTANT: Note that validation, testing and prediction phases depend on
    # hyperparameters collected during training phase. E.g. to run test phase,
    # it is therefor required to first setup "fit"

    datamodule.setup("fit")
    # datamodule.setup("validate")
    # datamodule.setup("test")
    # datamodule.setup("predict")

    # The whole training dataset may be accessed using
    train = datamodule.dataset["fit"]

    # Data module provides dataloaders for each subset. Make sure datamodule is correctly
    # initialized by calling above setup() routine.
    # Other available dataloaders are:
    # - val_dataloader
    # - test_dataloader
    # - predict_dataloader
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    # The returned dataloader is a vanilla torch.utils.data.DataLoader object
    # that shuffles samples for the training set. Samples are returned in batches;
    # the batch size is injected from config (see ./config/datamodule/multicell.yaml)
    i = 0
    best_corr = -1

    for epoch in range(100):

        print('Starting epoch ', epoch)

        for batch in train_dataloader:
            dnase, sequence, _, target, _ = batch.values()
            
            dnase = dnase.float().to(device)
            sequence = sequence.float().to(device)
            target = target.float().to(device)

            optimizer.zero_grad()

            output = model(torch.cat((dnase.unsqueeze(1), sequence), dim=1))
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                p_ctcff = np.corrcoef(output[:,0].cpu().detach().numpy().flatten(), target[:,0].cpu().detach().numpy().flatten())[0,1]
                p_ctcfr = np.corrcoef(output[:,1].cpu().detach().numpy().flatten(), target[:,1].cpu().detach().numpy().flatten())[0,1]
                p_sp1 = np.corrcoef(output[:,2].cpu().detach().numpy().flatten(), target[:,2].cpu().detach().numpy().flatten())[0,1]
                p_yy1 = np.corrcoef(output[:,3].cpu().detach().numpy().flatten(), target[:,3].cpu().detach().numpy().flatten())[0,1]
                p_avg = (p_ctcff + p_ctcfr + p_sp1 + p_yy1)/4

                print("Iter", str(i+1), "TRAIN (MSE):", str(loss.detach().item()), "Pearson AVG:", str(p_avg.item()))

            i = i+1

        # Validation
        with torch.no_grad():

            pearson_ctcff = 0.0
            pearson_ctcfr = 0.0
            pearson_yy1 = 0.0
            pearson_sp1 = 0.0


            o_all = []
            t_all = []
            for batch in val_dataloader:
                dnase_val, sequence_val, _, target_val, _ = batch.values()

                dnase_val = dnase_val.float().to(device)
                sequence_val = sequence_val.float().to(device)
                target_val = target_val.float().to(device)

                output = model(torch.cat((dnase_val.unsqueeze(1), sequence_val), dim=1))
                o_all.append(output.cpu().detach().numpy())
                t_all.append(target_val.cpu().detach().numpy())

            output = np.concatenate(o_all, axis=0)
            target_val = np.concatenate(t_all, axis=0)
            output_ctcff = output[:,0].flatten()
            output_ctcfr = output[:,1].flatten()
            output_sp1 = output[:,2].flatten()
            output_yy1 = output[:,3].flatten()

            target_ctcff = target_val[:,0].flatten()
            target_ctcfr = target_val[:,1].flatten()
            target_sp1 = target_val[:,2].flatten()
            target_yy1 = target_val[:,3].flatten()

            pearson_corr = np.corrcoef(output_ctcff, target_ctcff)[0,1]
            if np.isnan(pearson_corr):
                pearson_corr = 0
            pearson_ctcff += pearson_corr

            pearson_corr = np.corrcoef(output_ctcfr, target_ctcfr)[0,1]
            if np.isnan(pearson_corr):
                pearson_corr = 0
            pearson_ctcfr += pearson_corr

            pearson_corr = np.corrcoef(output_yy1, target_yy1)[0,1]
            if np.isnan(pearson_corr):
                pearson_corr = 0
            pearson_yy1 += pearson_corr

            pearson_corr = np.corrcoef(output_sp1, target_sp1)[0,1]
            if np.isnan(pearson_corr):
                pearson_corr = 0
            pearson_sp1 += pearson_corr

            val_corr = (pearson_ctcff + pearson_ctcfr + pearson_sp1 + pearson_yy1)/4

            if val_corr > best_corr:
                best_corr = val_corr
                os.makedirs(f"{save_path}", exist_ok=True)
                torch.save(model.state_dict(), os.path.join(f"{save_path}", f"model_stage1_checkpoint.pt"))
                print("best model: epoch {}, corr {}".format(epoch, val_corr))

    return

