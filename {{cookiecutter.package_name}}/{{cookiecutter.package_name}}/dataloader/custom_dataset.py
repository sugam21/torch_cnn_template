from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import os


class CustomDataset(Dataset):
    def __init__(
        self,
        image_data_path: str,
        is_train: bool = False,
        transform=None,
    ):
        self.image_data_path: str = image_data_path
        self.is_train: bool = is_train
        self.transform = transform

    def __len__(self) -> int:
        # logic to compute the length of entire dataset
        ...

    def __getitem__(self, idx: int) -> tuple:
        # logic to get image and label provided the index.
        ...
