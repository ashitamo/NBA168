import pandas as pd
from TeamNameDict import eng_to_s
import os
'''
將原始資料中的每支球隊的比賽資料分開存檔
'''

# 讀取 NBA_Game_Data.csv
file_path = "NBA_Game_Data.csv"  # 修改為你的檔案路徑
data = pd.read_csv(file_path)

# 將所有欄位名稱轉換為全大寫
data.columns = data.columns.str.upper()

# 將主客隊資料名稱轉換為大寫後進行映射
data['VISITOR_TEAM'] = data['VISITOR_TEAM'].str.upper()
data['HOME_TEAM'] = data['HOME_TEAM'].str.upper()

# 將主客隊資料合併並加上標籤
data['VISITOR_ABBREV'] = data['VISITOR_TEAM'].map(eng_to_s)
data['HOME_ABBREV'] = data['HOME_TEAM'].map(eng_to_s)

# 建立每支球隊的資料夾
output_folder = "teams_data"
os.makedirs(output_folder, exist_ok=True)

# 將資料分開並存檔
teams = set(eng_to_s.values())
for team in teams:
    team_data = data[(data['VISITOR_ABBREV'] == team) | (data['HOME_ABBREV'] == team)]
    if not team_data.empty:
        output_path = os.path.join(output_folder, f"{team}.csv")
        team_data.to_csv(output_path, index=False)
        print(f"資料已儲存: {output_path}")
