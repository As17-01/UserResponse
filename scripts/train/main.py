import pathlib
import sys

import hydra
import numpy as np
import omegaconf
import pandas as pd
from hydra_slayer import Registry
from loguru import logger
from sklearn.metrics import ndcg_score
from tqdm import tqdm

sys.path.append("../../")

import src

FEATURES = ["session_id", "recommendation_idx", "timestamp", "user_id", "item_id"]
TARGET = "response"


def calculate_score(data):
    score = 0

    for _, slate in tqdm(data.groupby(["recommendation_idx"])):
        score += ndcg_score(slate[TARGET].values[None, :], slate["pred"].values[None, :])

    return score / len(np.unique(data["recommendation_idx"]))


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: omegaconf.DictConfig) -> None:
    train_path = pathlib.Path(cfg.train_key)
    test_path = pathlib.Path(cfg.test_key)
    save_path = pathlib.Path(cfg.save_key)

    train_data = pd.read_csv(train_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)

    # cfg_dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    # registry = Registry()
    # registry.add_from_module(src, prefix="src.")
    # algorithm = registry.get_from_params(**cfg_dct["algorithm"])

    np.random.seed(100509)

    timestamps = np.unique(train_data.timestamp)
    is_train = train_data.timestamp < timestamps[int(len(timestamps) * 0.85)]

    train = train_data[is_train].copy()
    val = train_data[~is_train].copy()

    logger.info(f"DATA SIZE: {len(train_data)}")
    logger.info(f"TRAIN SIZE: {len(train)}")
    logger.info(f"VAL SIZE: {len(val)}")

    logger.info("Training...")

    # algorithm.fit(train[FEATURES], train[TARGET])

    train["pred"] = train[TARGET].values
    val["pred"] = val[TARGET].values

    train_score = calculate_score(train)
    val_score = calculate_score(val)

    logger.info(f"Train NDCG: {train_score}")
    logger.info(f"Val NDCG: {val_score}")

    logger.info("Predicting...")
    pass


if __name__ == "__main__":
    main()
