import json
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from logger import setup_logging
import os
import tomllib


class Config:
    """Takes the config file path and returns a Config object while setting up logger."""

    def __init__(
        self, data: dict[str, any], train: dict[str, any], model: dict[str, any]
    ) -> None:
        self.data: dict[str, any] = data
        self.train: dict[str, any] = train
        self.model: dict[str, any] = model
        setup_logging(save_dir=self.train["log_save_dir"])

    @classmethod
    def from_json(cls, config_path: str):
        check_file_if_exists(config_path)

        with open(config_path, mode="r") as file:
            config: dict[str, any] = json.load(file)
        return cls(config["data"], config["train"], config["model"])

    @classmethod
    def from_toml(cls, config_path: str):
        with open(config_path, mode="rb") as file:
            config: dict[str, any] = tomllib.load(file)
        return cls(config["data"], config["train"], config["model"])


def seed_everything(seed: int = 42) -> None:
    """
    This function seeds everything as name suggests
    PARAMS:
    seed (int): number to seed to

    RETURNS:
    none
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_image(loader, num_batch_to_show: int = 2) -> None:
    """TODO: docstring"""
    counter: int = 1
    for image_batch, label_batch in loader:
        fig, axes = plt.subplots(6, 5, figsize=(25, 8))
        # train_features, train_labels = next(iter(loader))
        for i, ax in enumerate(axes.flatten()):
            img = image_batch[i].permute(1, 2, 0)
            label = label_batch[i].item()
            ax.imshow(img)
            ax.set_title(label)
            ax.set_axis_off()
            print(f"Feature batch shape: {image_batch.size()}")
            print(f"Labels batch shape: {label_batch.size()}")
        plt.tight_layout()
        plt.show()
        counter += 1
        if counter == num_batch_to_show:
            break


def check_dir_if_exists(path: str) -> None:
    """Check if a certain dir exists or not"""
    assert os.path.isdir(path), f"{path} does not exists."


def check_file_if_exists(path: str) -> None:
    """Check if a certain file exists or not"""
    assert os.path.isfile(path), f"{path} file does not exists"


def create_dir(path: str) -> None:
    """Create a new directory if it does not exist."""
    os.mkdir(path)
