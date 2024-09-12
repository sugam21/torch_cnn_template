import pandas as pd
from torch.utils.data import DataLoader

from {{cookiecutter.package_name}}.base.base_dataloader import BaseDataLoader
from {{cookiecutter.package_name}}.utils import get_logger

from .custom_dataset import CustomDataset
from .data_transform import DataTransform

LOG: any = get_logger("dataloader")


class CustomDataLoader(BaseDataLoader):
    def __init__(self, data_path: dict[str, any]):
        super().__init__(data_path)

        LOG.debug(f"Loading the data from {self.data_path['image_path']}.... ")

        self.train, self.test = self._get_splits(self.image_path)

    def _get_splits(self, image_data_path: str) -> any:
        # train_size: float = 0.8
        # test_size: float = 1 - train_size
        # LOG.debug(
        #     f"Splitting the data into training: {train_size:.2f} testing: {test_size:.2f}"
        # )
        # Write the logic to split the data into training and testing. OR if you already have splitted data then just return the splits
        ...

    def get_train_dataloader(self, batch_size: int = 32) -> DataLoader:
        self._train_dataset = CustomDataset(
            image_data_path=self.data_path["image_path"],
            is_train=True,
            transform=DataTransform(input_size=self.data_path["image_size"]),
        )

        self.train_dataloader = DataLoader(
            self._train_dataset, batch_size=batch_size, shuffle=True
        )
        return self.train_dataloader

    def get_test_dataloader(self, batch_size: int = 32) -> DataLoader:
        self._test_dataset = CustomDataset(
            image_data_path=self.data_path["image_path"],
            image_label_df=self.test,
            is_train=False,
            transform=DataTransform(input_size=self.data_path["image_size"]),
        )

        self.test_dataloader = DataLoader(self._test_dataset, batch_size=batch_size)
        return self.test_dataloader
