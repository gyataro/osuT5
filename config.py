from functools import lru_cache
import logging
import yaml

CONFIG_PATH = "./config.yaml"


@lru_cache()
def load(path=CONFIG_PATH):
    try:
        config = yaml.safe_load(open(path))
    except Exception as error:
        logging.exception(f"failed to load config, reason: {error}")
    logging.info(f"loaded config from: {path}")
    return config
