import pandas as pd
from datetime import datetime

# Load the CSV data into a DataFrame
file_path = r'C:\Users\audi9\Desktop\NBA168-master\NBA_Game_Data.csv'  # Replace with actual path if needed
data = pd.read_csv(file_path)

# Define team name mapping dictionary
team_mapping = {
    'ATLANTA HAWKS': 'ATL', 'ST. LOUIS HAWKS': 'SLH', 'MILWAUKEE HAWKS': 'MIL', 
    'TRI-CITIES BLACKHAWKS': 'TCB', 'BOSTON CELTICS': 'BOS', 'BROOKLYN NETS': 'BKN', 
    'NEW JERSEY NETS': 'NJN', 'CHICAGO BULLS': 'CHI', 'CHARLOTTE HORNETS': 'CHA', 
    'CHARLOTTE BOBCATS': 'CHA', 'CLEVELAND CAVALIERS': 'CLE', 'DALLAS MAVERICKS': 'DAL', 
    'DENVER NUGGETS': 'DEN', 'DETROIT PISTONS': 'DET', 'FORT WAYNE PISTONS': 'FWP', 
    'GOLDEN STATE WARRIORS': 'GSW', 'SAN FRANCISCO WARRIORS': 'SFW', 
    'PHILADELPHIA WARRIORS': 'PHI', 'HOUSTON ROCKETS': 'HOU', 'INDIANA PACERS': 'IND', 
    'LOS ANGELES CLIPPERS': 'LAC', 'SAN DIEGO CLIPPERS': 'SDC', 'BUFFALO BRAVES': 'BUF', 
    'LOS ANGELES LAKERS': 'LAL', 'MINNEAPOLIS LAKERS': 'MIN', 'MEMPHIS GRIZZLIES': 'MEM', 
    'VANCOUVER GRIZZLIES': 'VAN', 'MIAMI HEAT': 'MIA', 'MILWAUKEE BUCKS': 'MIL', 
    'MINNESOTA TIMBERWOLVES': 'MIN', 'NEW ORLEANS PELICANS': 'NOP', 
    'NEW ORLEANS/OKLAHOMA CITY HORNETS': 'NOK', 'NEW ORLEANS HORNETS': 'NOH', 
    'NEW YORK KNICKS': 'NYK', 'OKLAHOMA CITY THUNDER': 'OKC', 'SEATTLE SUPERSONICS': 'SEA', 
    'ORLANDO MAGIC': 'ORL', 'PHILADELPHIA 76ERS': 'PHI', 'SYRACUSE NATIONALS': 'SYR', 
    'PHOENIX SUNS': 'PHX', 'PORTLAND TRAIL BLAZERS': 'POR', 'SACRAMENTO KINGS': 'SAC', 
    'KANSAS CITY KINGS': 'KCK', 'KANSAS CITY-OMAHA KINGS': 'KCK', 
    'CINCINNATI ROYALS': 'CIN', 'ROCHESTER ROYALS': 'ROR', 'SAN ANTONIO SPURS': 'SAS', 
    'TORONTO RAPTORS': 'TOR', 'UTAH JAZZ': 'UTA', 'NEW ORLEANS JAZZ': 'NOJ', 
    'WASHINGTON WIZARDS': 'WAS', 'WASHINGTON BULLETS': 'WAS', 'CAPITAL BULLETS': 'CAP', 
    'BALTIMORE BULLETS': 'BAL', 'CHICAGO ZEPHYRS': 'CHI', 'CHICAGO PACKERS': 'CHI', 
    'ANDERSON PACKERS': 'AND', 'CHICAGO STAGS': 'CHI', 'INDIANAPOLIS OLYMPIANS': 'IND', 
    'SHEBOYGAN RED SKINS': 'SRS', 'ST. LOUIS BOMBERS': 'SLB', 
    'WASHINGTON CAPITOLS': 'WAS', 'WATERLOO HAWKS': 'WAT', 'SAN DIEGO ROCKETS': 'SDR',
}

# Step 1: Convert Date and Start columns to a single DateTime column
data['Date'] = data['Date'] + ' ' + data['Start (ET)']
data['Date'] = pd.to_datetime(data['Date'])

# Step 2: Standardize and map team names to abbreviations
data['Visitor_Team'] = data['Visitor_Team'].str.upper().map(team_mapping)
data['Home_Team'] = data['Home_Team'].str.upper().map(team_mapping)

# Step 3: Drop 'visitor_USG%' and 'home_USG%' columns
data = data.drop(columns=['visitor_USG%', 'home_USG%','Start (ET)'])

# Step 4: Add a new column to determine the winning team
data['Home_Win'] = (data['Home_PTS'] > data['Visitor_PTS']).astype(int)

# Save the processed DataFrame to verify results
processed_file_path = 'processed_data.csv'
data.to_csv(processed_file_path, index=False)

