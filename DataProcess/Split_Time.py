import os
import pandas as pd

# 設定資料夾路徑
data_folder = "teams_data_restructured"
train_folder = "train_data"
val_folder = "val_data"
os.makedirs(train_folder, exist_ok=True)  # 如果資料夾不存在，則建立
os.makedirs(val_folder, exist_ok=True)    # 如果資料夾不存在，則建立

# 2024 年 11 月的篩選條件
cutoff_date = pd.Timestamp("2024-10-1")

# 遍歷資料夾中的每個文件
for file_name in os.listdir(data_folder):
    if file_name.endswith("_restructured.csv"):  # 確保只處理相關的 CSV 文件
        file_path = os.path.join(data_folder, file_name)
        
        # 讀取 CSV 文件
        data = pd.read_csv(file_path, parse_dates=["DateTime"])
        
        # 篩選 2024 年 11 月之後的比賽作為驗證集
        validation_data = data[data["DateTime"] >= cutoff_date]
        
        # 篩選 2024 年 11 月前的比賽作為訓練集
        train_data = data[data["DateTime"] < cutoff_date]
        
        # 儲存到對應的資料夾
        if not train_data.empty:
            train_data.to_csv(os.path.join(train_folder, file_name), index=False)
            print(f"訓練集已儲存：{os.path.join(train_folder, file_name)}")
        else:
            print(f"{file_name} 無符合條件的訓練集資料。")
        
        if not validation_data.empty:
            validation_data.to_csv(os.path.join(val_folder, file_name), index=False)
            print(f"驗證集已儲存：{os.path.join(val_folder, file_name)}")
        else:
            print(f"{file_name} 無符合條件的驗證集資料。")

print("資料分割完成！訓練集與驗證集已儲存。")
