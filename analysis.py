import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 讀取所有檔案並合併
def load_and_merge_data(folder_path):
    """
    讀取資料夾中所有 CSV 檔案並合併成一個 DataFrame。
    """
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    data_list = [pd.read_csv(file) for file in all_files]
    merged_data = pd.concat(data_list, ignore_index=True)
    merged_data = merged_data.sort_values('DateTime')  # 確保按比賽日期排序
    return merged_data

# 添加過去比賽數據作為滾動特徵
def add_past_game_features(data, feature_cols, num_past_games=3):
    """
    使用過去比賽的數據作為特徵，而不是計算平均值。
    """
    for col in feature_cols:
        for i in range(1, num_past_games + 1):
            shifted_col_name = f'{col}_past_game_{i}'
            # 將數據平移 i 場比賽
            data[shifted_col_name] = data.groupby('Team')[col].shift(i)
    
    # 移除缺失值（因為平移操作會導致 NaN）
    data = data.dropna()
    return data

# 訓練模型並分析結果
def train_and_evaluate(data, feature_cols, num_past_games=3, target_col='Is_Win'):
    """
    訓練模型，評估結果並繪製特徵重要性
    """
    # 準備特徵和目標變數
    past_game_feature_cols = [
        f'{col}_past_game_{i}'
        for col in feature_cols
        for i in range(1, num_past_games + 1)
    ]
    X = data[past_game_feature_cols]
    y = data[target_col]
    
    # 分割資料
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 訓練模型
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # 預測結果
    y_pred = model.predict(X_test)
    
    # 評估模型
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # 繪製特徵重要性
    plot_feature_importance(model, past_game_feature_cols)

def plot_feature_importance(model, feature_cols):
    """
    繪製特徵重要性
    """
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(feature_cols)), importances[indices], align="center")
    plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in indices], rotation=90)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

# 主程式
if __name__ == "__main__":
    # 資料夾路徑（替換為你的資料夾路徑）
    folder_path = "teams_data_restructured"
    
    # 1. 讀取並合併所有檔案
    data = load_and_merge_data(folder_path)
    
    # 2. 確認要使用的欄位
    feature_cols = [col for col in data.columns if col.startswith('Team_')]
    
    # 3. 添加過去比賽數據作為滾動特徵
    num_past_games = 6 # 可自由調整使用的過去比賽數量
    data = add_past_game_features(data, feature_cols, num_past_games=num_past_games)
    
    # 4. 訓練模型並分析結果
    train_and_evaluate(data, feature_cols, num_past_games=num_past_games)
