from typing import List
from glob import glob
import os
import argparse

import dotenv
import hydra
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
import pyarrow as pa

import numpy as np
import dask
import dask.dataframe as dd
import pandas as pd

from src.datamodules.encode_datamodule import MultiCellModule
from src.datasets.encode_dataset import CellDataset
from src import utils

def store_input_data(cell_line: str):
    # load environment variables from `.env` file if it exists
    # recursively searches for `.env` in all folders starting from work dir
    dotenv.load_dotenv(override=True)

    with initialize(version_base=None, config_path="configs", job_name="test_app"):
        config = compose(config_name="demo", return_hydra_config=True) #return_hydra_config=True

    HydraConfig.instance().set_config(config)

    log = utils.get_logger(__name__)

    # Init datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: MultiCellModule = hydra.utils.instantiate(
        config.datamodule, _recursive_=False
    )
    datamodule.setup("fit")
    datamodule.setup("predict")

    predict_dataloader = datamodule.predict_dataloader()

    datasets: List[CellDataset] = predict_dataloader.dataset.datasets

    dataset = datasets[0]
    targets: pd.DataFrame = dataset.targets.copy()
    targets = targets.reset_index(drop=True)

    log.info("Prepare dataset for prediction")

    # Store the original index
    targets["dataset_index"] = targets.index
    targets[["dnase", "sequence", "mappability"]] = None
    targets = targets.sort_values(by="chrom")

    # Create output folder
    output_dir = os.path.join(
        hydra.utils.get_original_cwd(), f"data/stage1_outputs/predict_{cell_line.lower()}"
    )
    os.makedirs(output_dir)

    log.info("Init dask task graph")
    ddf = dd.from_pandas(targets, chunksize=2**12)


    # Extract features from dataset
    @dask.delayed
    def features(df):
        item = dataset[df.dataset_index]
        df["dnase"] = item["dnase"].numpy().tolist()

        a = item["sequence"].numpy().flatten().tolist()
        df["sequence"] = a
        
        b = item["mappability"].numpy().flatten()
        b[np.isnan(b)]=0
        b = b.tolist()
        df["mappability"] = b
        
        return df

    ddf = ddf.apply(features, axis=1, meta=targets)

    # Predict in batches
    @dask.delayed
    def predict(df):
        return df

    predictions = dd.from_delayed(
        [predict(partition) for partition in ddf.partitions], meta=targets
        )

    expected_schema = pa.schema([
        ('species', pa.string()),
        ('cell_line', pa.string()),
        ('chrom', pa.string()),
        ('bin', pa.int64()),
        ('atacpeak', pa.float64()),
        ('chromStart', pa.int64()),
        ('chromEnd', pa.int64()),
        ('center', pa.int64()),
        ('bin_start', pa.int64()),
        ('bin_end', pa.int64()),
        ('original_index', pa.int64()),
        ('dataset_index', pa.int64()),
        ('dnase', pa.list_(pa.float64())),
        ('sequence', pa.list_(pa.float64())),
        ('mappability', pa.list_(pa.float64()))
    ])

    log.info("Write intermediate results to disk")
    predictions.to_parquet(
        f"{output_dir}/tmp", schema=expected_schema, engine="pyarrow", write_index=False, partition_on=["chrom"]
    )

    log.info("Clean up memory")
    del predict_dataloader, datasets, dataset, targets

    log.info("Load intermediate results from disk")

    for partition in glob(f"{output_dir}/tmp/chrom=*"):
        chrom = partition.split("=")[1]

        log.info(f"Merge files for {chrom}")
        parts = dd.read_parquet(partition).repartition(npartitions=1)
        log.info(
            f"Memory usage per partion: {parts.memory_usage_per_partition(deep=True)}"
        )

        # Bring back chrom info
        parts["chrom"] = chrom

        # Write partition to disk
        # If multiple partitions are created, append index to file name e.g. chr1-1
        parts.to_parquet(
            output_dir,
            name_function=lambda x: f"{chrom}{'' if x == 0 else '-'}{'' if x == 0 else x}.pq",
            engine="pyarrow",
            write_index=False,
            schema=expected_schema
        )

    # Delete tmp folder
    os.system(f"rm -rf {output_dir}/tmp")

    # Make sure everything closed properly
    log.info("Finalizing!")

def argparser():
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("--cell_line", type=str, help="Cell line to predict", required=True)
    return parser.parse_args()

def main():
    args = argparser()
    store_input_data(args.cell_line)

if __name__ == "__main__":
    main()
