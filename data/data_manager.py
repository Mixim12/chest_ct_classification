from torch.utils.data import DataLoader

from data.base_dataset import BaseDataset
from data.base_dataset_test import BaseDatasetTest


class DataManager:
    def __init__(self, config):
        self.config = config
        self.batch_size = self.config["batch_size"]

    def get_train_valid_test_dataloaders(self, num_workers=4):
        train_dataset = BaseDataset(path=self.config["train_path"], mode="train")
        valid_dataset = BaseDataset(path=self.config["valid_path"], mode="valid")
        test_dataset = BaseDatasetTest(path=self.config["test_path"])

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        validation_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, validation_loader, test_loader
