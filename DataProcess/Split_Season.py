import pandas as pd

# 載入資料
file_path = r'processed_data.csv'
data = pd.read_csv(file_path)

# 將 DATE 轉換為 datetime 格式
data['DATE'] = pd.to_datetime(data['DATE'])

# 增加 SEASON 欄位以標記賽季
def assign_season(date):
    year = date.year
    if date.month >= 10:  # 從10月開始為新賽季
        return f"{year}-{year + 1}"
    else:  # 從1月至6月
        return f"{year - 1}-{year}"

data['SEASON'] = data['DATE'].apply(assign_season)

# 根據賽季分隔並儲存 CSV
seasons = data['SEASON'].unique()
for season in seasons:
    season_data = data[data['SEASON'] == season]
    season_file = f'Season_Data/{season}_season.csv'
    season_data.to_csv(season_file, index=False)
    print(f"{season} 賽季資料已儲存到 {season_file}")
