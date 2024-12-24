from typing import Optional
import argparse

import hydra
from omegaconf import DictConfig

from src import utils
from src.datamodules.encode_datamodule import MultiCellModule

from src.models.trepii_model import DeepC

import numpy as np
import torch

import os

def get_logger():
    log = utils.get_logger(__name__)
    return log

def parse_args():
    parser = argparse.ArgumentParser(description='Test a model')
    parser.add_argument('--model_path', type=str, help='path to load the model', required=True)
    return parser.parse_args()

def run(config: DictConfig) -> Optional[float]:
    args = parse_args()
    model_path = args.model_path
    
    log = get_logger()

    # Load model
    model = DeepC()
    weights = torch.load(model_path, map_location='cpu')
    model.load_state_dict(weights)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

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
    datamodule.setup("test")

    # Data module provides dataloaders for each subset. Make sure datamodule is correctly 
    # initialized by calling above setup() routine. 
    predict_dataloader = datamodule.test_dataloader()


    # The returned dataloader is a vanilla torch.utils.data.DataLoader object
    # that shuffles samples for the training set. Samples are returned in batches; 
    # the batch size is injected from config (see ./config/datamodule/multicell.yaml)

    # Validation
    with torch.no_grad():

        pred_ctcff = []
        pred_ctcfr = []
        pred_yy1 = []
        pred_sp1 = []

        gt_ctcff = []
        gt_ctcfr = []
        gt_yy1 = []
        gt_sp1 = []

        for batch in predict_dataloader:
            dnase_val, sequence_val, _, target_val, _ = batch.values()

            dnase_val = dnase_val.float().to(device)
            sequence_val = sequence_val.float().to(device)
            target_val = target_val.float().to(device)

            output = model(torch.cat((dnase_val.unsqueeze(1), sequence_val), dim=1))
            output_ctcff = output[:,0].cpu().detach().numpy().flatten()
            pred_ctcff.append(output_ctcff)
            output_ctcfr = output[:,1].cpu().detach().numpy().flatten()
            pred_ctcfr.append(output_ctcfr)
            output_sp1 = output[:,2].cpu().detach().numpy().flatten()
            pred_sp1.append(output_sp1)
            output_yy1 = output[:,3].cpu().detach().numpy().flatten()
            pred_yy1.append(output_yy1)

            target_ctcff = target_val[:,0].cpu().numpy().flatten()
            gt_ctcff.append(target_ctcff)
            target_ctcfr = target_val[:,1].cpu().numpy().flatten()
            gt_ctcfr.append(target_ctcfr)
            target_sp1 = target_val[:,2].cpu().numpy().flatten()
            gt_sp1.append(target_sp1)
            target_yy1 = target_val[:,3].cpu().numpy().flatten()
            gt_yy1.append(target_yy1)

        pearson_corr = np.corrcoef(np.concatenate(pred_ctcff, axis=0).flatten(), np.concatenate(gt_ctcff, axis=0).flatten())[0,1]
        print(pearson_corr)
        pearson_corr = np.corrcoef(np.concatenate(pred_ctcfr, axis=0).flatten(), np.concatenate(gt_ctcfr, axis=0).flatten())[0,1]
        print(pearson_corr)
        pearson_corr = np.corrcoef(np.concatenate(pred_yy1, axis=0).flatten(), np.concatenate(gt_yy1, axis=0).flatten())[0,1]
        print(pearson_corr)
        pearson_corr = np.corrcoef(np.concatenate(pred_sp1, axis=0).flatten(), np.concatenate(gt_sp1, axis=0).flatten())[0,1]
        print(pearson_corr)


    return

