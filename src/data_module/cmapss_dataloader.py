from torch.utils.data import DataLoader
from data_module.cmapss_dataset import CMAPSS_Dataset


class FullDataLoader:
    def __init__(self, data_dir: str = "data", window: int = 5, batch_size: int = 64):
        train_dset = CMAPSS_Dataset(data_dir, "train", window)
        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)

        val_dset = CMAPSS_Dataset(data_dir, "val", window)
        self.val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False)

        test_dset = CMAPSS_Dataset(data_dir, "test", window)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader
