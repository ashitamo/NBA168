import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class NBA_Dataset(Dataset):
    def __init__(self, data_path, mean_std_file, window=10):
        self.data = pd.read_csv(data_path)
        self.mean_std = pd.read_csv(mean_std_file, index_col=0)
        self.window = window

    def preprocess(self, data):
        # 正規化數據
        for col in data.columns:
            if col in self.mean_std.index and col != 'Is_Win':
                mean = self.mean_std.loc[col, 'Mean']
                std = self.mean_std.loc[col, 'Std']
                if std != 0:
                    data[col] = (data[col] - mean) / std