import logging


def get_logger(name=__name__) -> logging.Logger:
    """Initializes a simple Python command line logger."""
    logger = logging.getLogger(name)

    # Set a default logging level if not already set
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger

log = get_logger(__name__)