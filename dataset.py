import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class NBA_Dataset(Dataset):
    def __init__(self, folder_path, mean_std_path, window=3):
        """
        初始化 NBA Dataset，確保滑窗操作僅應用於同一支隊伍內的數據。
        
        Args:
            folder_path (str): 資料夾路徑，包含多個 CSV 檔案。
            mean_std_path (str): 儲存 mean 和 std 的 CSV 路徑。
            window (int): 時窗大小，用於獲取連續比賽數據。
        """
        self.folder_path = folder_path
        self.window = window
        self.mean_std = pd.read_csv(mean_std_path).set_index('Column')
        self.s_to_id = {
            'BOS': 0, 'NJN': 1, 'NYK': 2, 'PHI': 3, 'TOR': 4, 'CHI': 5, 'CLE': 6,
            'DET': 7, 'IND': 8, 'MIL': 9, 'ATL': 10, 'CHA': 11, 'MIA': 12, 'ORL': 13,
            'WAS': 14, 'DAL': 15, 'HOU': 16, 'MEM': 17, 'NOP': 18, 'SAS': 19, 'DEN': 20,
            'MIN': 21, 'POR': 22, 'SEA': 23, 'UTA': 24, 'GSW': 25, 'LAC': 26, 'LAL': 27,
            'PHX': 28, 'SAC': 29, 'BKN': 30, 'OKC': 31
        }
        self.data, self.labels = self.load_and_prepare_data()
        self.features = [col for col in self.data.columns if col.startswith('Team_') or col.startswith('Opponent_')]

    def load_and_prepare_data(self):
        """
        載入資料夾內所有 CSV 檔案，分別處理每支隊伍的數據，並合併。
        
        Returns:
            pd.DataFrame: 滑窗後的數據。
            pd.Series: 滑窗後的標籤。
        """
        all_files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        data_list = []

        for file in all_files:
            data = pd.read_csv(file)
            team_name = os.path.basename(file).split('_')[0]  # 假設文件名格式為 "隊名_restructured.csv"

            

            # 正規化數據
            for col in data.columns:
                if col in self.mean_std.index and col != 'Is_Win':
                    mean = self.mean_std.loc[col, 'Mean']
                    std = self.mean_std.loc[col, 'Std']
                    if std != 0:
                        data[col] = (data[col] - mean) / std

            # 添加隊伍 ID
            data['Team_ID'] = self.normalize_team_id(self.s_to_id[team_name])
            data['Opponent_ID'] = self.normalize_team_id(data['Opponent'].map(self.s_to_id))
            
            # 移除不需要的欄位
            data = data.drop(columns=['DateTime', 'Team', 'Opponent',"Team_USG%","Opponent_USG%"], errors='ignore')

            # 滑窗處理
            team_data, team_labels = self.apply_sliding_window(data)
            data_list.append((team_data, team_labels))

        # 合併所有隊伍的數據
        all_data = pd.concat([item[0] for item in data_list], ignore_index=True)
        all_labels = pd.concat([item[1] for item in data_list], ignore_index=True)
        return all_data, all_labels


    def apply_sliding_window(self, team_data):
        # 原滑窗邏輯
        sliding_data = []
        sliding_labels = []
        
        for i in range(len(team_data) - self.window):
            window_data = team_data.iloc[i+1:i + self.window].copy()  # 時窗內數據（不含當前比賽）
            current_game = team_data.iloc[i]  # 當前比賽數據
            
            # 拉平成特徵並加入對手和隊伍 ID
            features = window_data.values.flatten().tolist()
            features += [
                self.normalize_team_id(current_game['Opponent_ID']),
                self.normalize_team_id(current_game['Team_ID']),
                current_game['Is_Home']
            ]
            
            # 當前比賽結果作為標籤
            label = current_game['Is_Win']
            sliding_data.append(features)
            sliding_labels.append(label)
        
        sliding_data_df = pd.DataFrame(sliding_data)
        sliding_data_df.columns = [f"Feature_{i}" for i in range(sliding_data_df.shape[1])]  # 為列名添加唯一標識符
        
        return sliding_data_df, pd.Series(sliding_labels)


    def normalize_team_id(self, team_id):
        """
        正規化隊伍 ID。
        
        Args:
            team_id (int): 隊伍 ID。
        
        Returns:
            float: 正規化後的隊伍 ID。
        """
        return (team_id - 16.0) / 32.0

    def __len__(self):
        """
        返回資料集的長度。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回單個樣本，包括特徵和標籤。

        Args:
            idx (int): 數據索引。

        Returns:
            torch.Tensor: 展平後的特徵。
            torch.Tensor: 標籤。
        """
        features = self.data.iloc[idx].values
        label = self.labels.iloc[idx]
        
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return feature_tensor, label_tensor


# 測試 Dataset 類別
if __name__ == "__main__":
    folder_path = "teams_data_restructured"  # 替換為你的資料夾路徑
    mean_std_path = "column_mean_and_std.csv"  # 替換為 mean/std 檔案路徑
    window_size = 2  # 可修改時窗大小

    dataset = NBA_Dataset(folder_path, mean_std_path, window=window_size)
    print("資料集大小:", len(dataset))
    print("特徵大小:", dataset[0][0].shape)
    print("標籤大小:", dataset[0][1].shape)
