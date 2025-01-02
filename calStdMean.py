import os
import pandas as pd

# 計算每個欄位的均值和標準差
def calculate_mean_and_std(data):
    """
    計算數據中每個數值型欄位的均值和標準差。
    """
    # 篩選數值型欄位
    numeric_cols = data.select_dtypes(include=['number']).columns
    stats = pd.DataFrame({
        'Mean': data[numeric_cols].mean(),
        'Std': data[numeric_cols].std()
    })
    return stats

def load_and_merge_data(folder_path):
    # 讀取所有 CSV 檔案
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    data = pd.concat([pd.read_csv(os.path.join(folder_path, f)) for f in csv_files], ignore_index=True)
    return data

# 主程式
if __name__ == "__main__":
    # 資料夾路徑（替換為你的資料夾路徑）
    folder_path = "teams_data_restructured"

    # 1. 讀取並合併所有檔案
    data = load_and_merge_data(folder_path)
    # 2. 計算每個欄位的均值和標準差
    stats = calculate_mean_and_std(data)
    
    # 3. 顯示或儲存結果
    print("每個數值欄位的均值和標準差：")
    print(stats)
    
    # 如果需要將結果保存為 CSV
    stats.to_csv("column_mean_and_std.csv", index_label="Column")
