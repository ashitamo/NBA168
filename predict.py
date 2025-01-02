import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from model import Net
from DataProcess.TeamNameDict import s_to_id,eng_to_s,ch_to_s

# 預測參數
Team = '休士頓火箭'
Opponent = '達拉斯獨行俠'
if Team not in s_to_id.keys():
    if Team.upper() in ch_to_s.keys():
        Team = ch_to_s[Team]
    else:
        Team = eng_to_s[Team.upper()]
if Opponent not in s_to_id.keys():
    if Opponent.upper() in ch_to_s.keys():
        Opponent = ch_to_s[Opponent]
    else:
        Opponent = eng_to_s[Opponent.upper()]
    
Is_Home = 1
date = '2024-12-28 15:00:00'
window = 7

def normalize(data):
    mean_std = pd.read_csv('column_mean_and_std.csv').set_index('Column')
    # 正規化數據
    # data = data.copy()
    for col in data.columns:
        if col in mean_std.index and col != 'Is_Win':
            mean = mean_std.loc[col, 'Mean']
            std = mean_std.loc[col, 'Std']
            if std != 0:
                data[col] = (data[col] - mean) / std
    return data

data = pd.read_csv('teams_data_restructured/{}_restructured.csv'.format(Team))
data['DateTime'] = pd.to_datetime(data['DateTime'])
data = data.sort_values(by='DateTime', ascending=False)
data = data[data['DateTime'] < date].head(window)
data = normalize(data)
data['Team'] = (data['Team'].map(s_to_id) - 14.5) / 29.0
data['Opponent'] = (data['Opponent'].map(s_to_id) - 14.5) / 29.0
feature = data.drop(columns=['DateTime'])
feature = feature.to_numpy().flatten()

TeamNorm = (s_to_id[Team] - 14.5) / 29.0
OpponentNorm = (s_to_id[Opponent] - 14.5) / 29.0

feature = np.append(feature,[TeamNorm,OpponentNorm,Is_Home])
feature = torch.tensor(feature, dtype=torch.float32)

print(feature.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(feature.shape[0]).to(device)
model.load_state_dict(torch.load('modelacc.pth',weights_only=True))
feature = feature.to(device)
model.eval()

with torch.no_grad():
    win_probability = model(feature.unsqueeze(0)).item()

# 輸出結果
print(f"比賽日期: {date}")
print(f"隊伍: {Team} 對戰 {Opponent}")
print(f"比賽結果: {Team if win_probability > 0.5 else Opponent}")
print(f"預測勝率: {win_probability:.2%}")
