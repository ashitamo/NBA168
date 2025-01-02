import pandas as pd
import os

'''
將原始資料夾中的 CSV 檔案重新組織成更加易於處理的格式。
'''

# 設定資料夾路徑
folder_path = "teams_data"  # 修改為你的資料夾路徑
output_folder = "teams_data_restructured"  # 設定輸出資料夾，修改為你的輸出資料夾路徑
os.makedirs(output_folder, exist_ok=True)  # 如果輸出資料夾不存在，則建立

# 讀取資料夾中的所有 CSV 檔案
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):  # 只處理 CSV 檔案
        file_path = os.path.join(folder_path, filename)
        
        # 讀取原始資料
        data = pd.read_csv(file_path)

        # 合併 Date 和 Start (ET) 成 DateTime
        data['DateTime'] = pd.to_datetime(data['DATE'] + ' ' + data['START (ET)'])

        # 判斷是否主場並新增 Is_Home
        data['Is_Home'] = (data['HOME_ABBREV'] == filename.split('.')[0]).astype(int)

        # 判斷勝負並新增 Is_Win
        data['Is_Win'] = ((data['Is_Home'] == 1) & (data['HOME_PTS'] > data['VISITOR_PTS'])) | \
                         ((data['Is_Home'] == 0) & (data['VISITOR_PTS'] > data['HOME_PTS']))
        data['Is_Win'] = data['Is_Win'].astype(int)

        # 將數據按照 Team 和 Opponent 分組，確保 Team_* 代表自己隊伍，Opponent_* 代表對手
        team_cols = ['PTS', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 
                     'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'TS%', 'EFG%', 
                     '3PAR', 'FTR', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 
                     'ORTG', 'DRTG']

        for col in team_cols:
            data[f'Team_{col}'] = data.apply(lambda row: row[f'HOME_{col}'] if row['Is_Home'] else row[f'VISITOR_{col}'], axis=1)
            data[f'Opponent_{col}'] = data.apply(lambda row: row[f'VISITOR_{col}'] if row['Is_Home'] else row[f'HOME_{col}'], axis=1)

        # 新增 Team 和 Opponent 欄位
        team_name = filename.split('.')[0]
        data['Team'] = team_name
        data['Opponent'] = data.apply(lambda row: row['VISITOR_ABBREV'] if row['Is_Home'] == 1 else row['HOME_ABBREV'], axis=1)

        # 選擇最終需要的欄位
        final_cols = ['DateTime', 'Team', 'Opponent', 'Is_Home', 'Is_Win'] + \
                     [f'Team_{col}' for col in team_cols] + \
                     [f'Opponent_{col}' for col in team_cols]

        final_data = data[final_cols]

        # 按照 DateTime 降冪排序 (最舊的資料會在底部，最新的在最上面)
        final_data = final_data.sort_values(by='DateTime', ascending=False)

        # 儲存修改後的檔案
        output_path = os.path.join(output_folder, f"{team_name}_restructured.csv")
        final_data.to_csv(output_path, index=False)

        print(f"資料已成功儲存至 {output_path}")
