import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
from DataProcess.TeamNameDict import s_to_id

class NBA_Dataset(Dataset):
    def __init__(self, data_folder, mean_std_file, window=3):
        self.window = window
        # Convert 'Date' column to datetime
        self.mean_std = pd.read_csv(mean_std_file).set_index('Column')
        self.teamData = {}
        for team in s_to_id.keys():
            self.teamData[team] = pd.read_csv(os.path.join(data_folder, team + '_restructured.csv'))
            self.teamData[team]['DateTime'] = pd.to_datetime(self.teamData[team]['DateTime'])
            self.teamData[team] = self.teamData[team].sort_values(by='DateTime', ascending=False)
            self.teamData[team] = self.normalize(self.teamData[team])
            self.teamData[team]['Team'] = (self.teamData[team]['Team'].map(s_to_id) - 14.5) / 29.0
            self.teamData[team]['Opponent'] = (self.teamData[team]['Opponent'].map(s_to_id) - 14.5) / 29.0
        
        

        self.preprocess()

    def normalize(self, data):
        # 正規化數據
        # data = data.copy()
        for col in data.columns:
            if col in self.mean_std.index and col != 'Is_Win':
                mean = self.mean_std.loc[col, 'Mean']
                std = self.mean_std.loc[col, 'Std']
                if std != 0:
                    data[col] = (data[col] - mean) / std
        return data

    def fetch_team_data(self,data,date):
        filtered_data = data[data['DateTime'] < date]
        return filtered_data.head(self.window)


    def __len__(self):
        return len(self.features)

    def preprocess(self):
        self.features = []
        self.labels = []
        for team in tqdm(self.teamData.keys()):   
            data = self.teamData[team]
            for idx in range(len(data)):
                feature = self.fetch_team_data(data,data['DateTime'][idx])
                if feature.shape[0] < self.window:
                    continue
                feature = feature.drop(columns=['DateTime'])
                Team = data['Team'][idx]
                Opponent = data['Opponent'][idx]
                Is_Home = data['Is_Home'][idx]
                
                temp = np.array([Team,Opponent,Is_Home])
                feature = feature.to_numpy().flatten()
                feature = np.append(feature,temp)
                label = data['Is_Win'][idx]

                self.features.append(feature)
                self.labels.append(label)

    def __getitem__(self, index):
        feature = torch.tensor(self.features[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32).unsqueeze(0)
        return feature, label
    
if __name__ == "__main__":
    data_folder = "teams_data_restructured"
    mean_std_file = "column_mean_and_std.csv"
    window = 6
    dataset = NBA_Dataset(data_folder, mean_std_file,window)
    print(dataset[0][0].shape, dataset[0][1].shape)
    for i in range(len(dataset)-100, len(dataset)):
        features, label = dataset[i]
        print(f"第{i}筆資料的特徵為:{features.shape},標籤為:{label}")
    print(dataset[0])