import pandas as pd
from datetime import datetime

def fetch_team_data(file_path, team, date, window):
    """
    Fetches the specified number of games (window) for the given team before the specified date.

    Args:
        file_path (str): Path to the CSV file containing game data.
        team (str): Team abbreviation.
        date (str): Date in 'YYYY-MM-DD' format.
        window (int): Number of games to fetch before the specified date.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered game data.
    """
    # Load data
    data = pd.read_csv(file_path)

    # Convert 'Date' column to datetime
    data['DATE'] = pd.to_datetime(data['DATE'])

    # Parse input date to datetime
    target_date = datetime.strptime(date, '%Y-%m-%d')

    # Filter rows where the team appears as either Visitor or Home and the game is before the target date
    filtered_data = data[((data['VISITOR_TEAM'] == team) | (data['HOME_TEAM'] == team)) & (data['DATE'] < target_date)]

    # Sort by Date descending to get the most recent games first
    filtered_data = filtered_data.sort_values(by='DATE', ascending=False)

    # Return the specified number of rows
    return filtered_data.head(window)

# Example usage
file_path = r"C:\Users\audi9\Desktop\NBA168\processed_data.csv"
team = "LAL"  # Example team abbreviation
date = "2024-11-01"  # Example date
window = 10  # Example number of games to fetch

result = fetch_team_data(file_path, team, date, window)
print(result)
