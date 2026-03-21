import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CMAPSS_Dataset(Dataset):
    def __init__(self, data_dir: str, dset_type: str = "train", window: int = 5):
        dset_dir = f"{data_dir}/cmapss_{dset_type}.csv"
        self.df = pd.read_csv(dset_dir)
        self.window = window

        self.input_features = []
        self.meta_features = []
        self._assign_features()

        self.inputs = self.df[self.input_features].values.astype(np.float32)
        self.metas = self.df[self.meta_features].values.astype(np.float32)

        self.samples = []
        self._group_samples_by_unit()

    def _assign_features(self):
        for i in range(3):
            self.input_features.append(f"ops_setting_{i + 1}")
        for i in range(21):
            self.input_features.append(f"sensor_{i + 1}")

        self.meta_features = [
            "unit",
            "cycle",
            "engine",
            "rul",
            "health_idx",
            "health_level",
        ]

    def _group_samples_by_unit(self):
        for _, unit_df in self.df.groupby(["engine", "unit"], sort=False):
            unit_idx = unit_df.index.to_numpy()
            if len(unit_idx) < self.window:
                continue
            for start in range(len(unit_idx) - self.window + 1):
                self.samples.append(unit_idx[start : start + self.window])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rows = self.samples[idx]
        x = torch.from_numpy(self.inputs[rows])
        meta = torch.from_numpy(self.metas[rows])
        return x, meta


if __name__ == "__main__":
    data_dir = "data"
    dataset = CMAPSS_Dataset(data_dir)
    item, meta_data = dataset.__getitem__(0)
    print(item.shape, len(meta_data), meta_data[0])
    print(item[0])
