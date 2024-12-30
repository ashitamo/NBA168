import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from model import Net
from DataProcess.TeamNameDict import s_to_id,eng_to_s

# 預測參數
Team = 'TOR'
Opponent = 'MEM'
Is_Home = 0
if Team not in s_to_id.keys():
    Team = eng_to_s[Team.upper()]
if Opponent not in s_to_id.keys():
    Opponent = eng_to_s[Opponent.upper()]

date = '2024-12-26 7:30:00'
window = 1  # 使用前7場比賽數據作為輸入

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = Net(399).to(device)
model.load_state_dict(torch.load("modelacc.pth",weights_only=True))
model.eval()

# 載入資料
team_data_path = f"teams_data_restructured/{Team}_restructured.csv"
mean_std_path = "column_mean_and_std.csv"
mean_std = pd.read_csv(mean_std_path).set_index('Column')

# 載入隊伍資料
team_data = pd.read_csv(team_data_path)

# 將 DateTime 字串轉換為 datetime 格式
team_data['DateTime'] = pd.to_datetime(team_data['DateTime'])

# 過濾出指定日期之前的數據
game_date = pd.to_datetime(date)
filtered_data = team_data[team_data['DateTime'] < game_date]

# 確保有足夠的歷史比賽數據
if len(filtered_data) < window:
    raise ValueError(f"Not enough games before {date} to create a window of size {window}")

# 提取最近的 window 場比賽
recent_games = filtered_data.iloc[:window]

# 正規化數據
for col in recent_games.columns:
    if col in mean_std.index and col not in ['Is_Win', 'DateTime']:
        mean = mean_std.loc[col, 'Mean']
        std = mean_std.loc[col, 'Std']
        if std != 0:
            recent_games.loc[:, col] = (recent_games[col] - mean) / std  # 使用 .loc 明確指定列
# 正規化 Opponent_ID 和 Team_ID
def normalize_team_id(team_id):
    return (team_id - 16) / 32

# 對 recent_games 建立副本
recent_games = recent_games.copy()

# 添加對手和隊伍 ID
recent_games['Opponent_ID'] = normalize_team_id(recent_games['Opponent'].map(s_to_id))
recent_games['Team_ID'] = normalize_team_id(recent_games['Team'].map(s_to_id))

recent_games = recent_games.drop(columns=['DateTime', 'Team', 'Opponent',"Team_USG%","Opponent_USG%"], errors='ignore')

# 確保隊伍與對手 ID 存在
if recent_games['Opponent_ID'].isna().any() or recent_games['Team_ID'].isna().any():
    raise ValueError("Mapping for Team or Opponent is incomplete. Check the 's_to_id' dictionary.")



opponent_id = s_to_id[Opponent]
normalized_opponent_id = normalize_team_id(opponent_id)
team_id = s_to_id[Team]
normalized_team_id = normalize_team_id(team_id)

# 構建輸入特徵
features = [col for col in recent_games.columns if col.startswith('Team_') or col.startswith('Opponent_')]
feature_window = recent_games[features].values.flatten()
Is_Home = (Is_Home-0.5)/0.5
input_features = list(feature_window) + [normalized_opponent_id, normalized_team_id, Is_Home]

# 轉為 PyTorch Tensor
input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device)

# 預測勝率
with torch.no_grad():
    win_probability = model(input_tensor).item()

# 輸出結果
print(f"比賽日期: {date}")
print(f"隊伍: {Team} 對戰 {Opponent}")
print(f"預測勝率: {win_probability:.2%}")
