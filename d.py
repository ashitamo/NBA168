import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from datetime import datetime

class NBA_Dataset(Dataset):
    def __init__(self, data_path, mean_std_file, window=10):
        self.data = pd.read_csv(data_path)
        self.mean_std = pd.read_csv(mean_std_file, index_col=0)
        self.window = window

    def fetch_team_data(self,data, team, date, window):
        """
        Fetches the specified number of games (window) for the given team before the specified date.

        Args:
            file_path (str): Path to the CSV file containing game data.
            team (str): Team abbreviation.
            date (str): Date in 'YYYY-MM-DD' format.
            window (int): Number of games to fetch before the specified date.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered game data.
        """

        # Convert 'Date' column to datetime
        data['DATE'] = pd.to_datetime(data['DATE'])

        # Parse input date to datetime
        target_date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

        # Filter rows where the team appears as either Visitor or Home and the game is before the target date
        filtered_data = data[((data['VISITOR_TEAM'] == team) | (data['HOME_TEAM'] == team)) & (data['DATE'] < target_date)]

        # Sort by Date descending to get the most recent games first
        filtered_data = filtered_data.sort_values(by='DATE', ascending=False)

        # Return the specified number of rows
        return filtered_data.head(window)


    def preprocess(self, data):
        # 正規化數據
        for col in data.columns:
            if col in self.mean_std.index and col != 'Is_Win':
                mean = self.mean_std.loc[col, 'Mean']
                std = self.mean_std.loc[col, 'Std']
                if std != 0:
                    data[col] = (data[col] - mean) / std
        # 添加隊伍 ID
        data['VISITOR_TEAM'] = (data['VISITOR_TEAM'].map(self.s_to_id)-16.0 )/ 32.0
        data['HOME_TEAM'] = (data['HOME_TEAM'].map(self.s_to_id)-16.0 )/ 32.0

        return data

    def __len__(self):
        return len(self.data)-self.window

    def __getitem__(self, index):
        featuresTeam1 = self.fetch_team_data(self.data, self.data.iloc[index]['VISITOR_TEAM'], self.data.iloc[index]['DATE'], self.window)
        featuresTeam2 = self.fetch_team_data(self.data, self.data.iloc[index]['HOME_TEAM'], self.data.iloc[index]['DATE'], self.window)
        features = pd.concat([featuresTeam1, featuresTeam2], axis=1)
        features = self.preprocess(features)
        features = features.flatten().tolist()
        label = self.data.iloc[index]['HOME_WIN']

        return features, label
    
if __name__ == "__main__":
    data_path = "processed_data.csv"
    mean_std_file = "column_mean_and_std.csv"
    window = 10
    dataset = NBA_Dataset(data_path, mean_std_file,window)
    print(dataset[0])