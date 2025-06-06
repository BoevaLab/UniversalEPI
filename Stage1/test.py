import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="demo.yaml")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.test_pipeline import run

    # Applies optional utilities
    utils.extras(config)
    
    # Train model
    metric = run(config)

    print(
        "-------------------------------- End of pipeline --------------------------------"
    )

    return metric


if __name__ == "__main__":
    main()
