import yaml
from bunch import Bunch


def getConfig(file):
    """
    Get the config from a yaml file.
    Adds the default value for missing parameters
    in the configuration file.

    Args:
        file: yaml file containing the configuration
    Returns:
        config: Bunch object.
    """
    # parse the configurations from the config json file provided
    with open(file, "r") as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.Loader)
    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config
