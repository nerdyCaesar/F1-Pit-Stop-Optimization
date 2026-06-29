import requests
import pandas as pd
import numpy as np


# the bulk api endpoint now requires an API key due to heavy load on the servers 
# so we will be using the free livestream API endpoint which I have used here which bypasses the need for an API key and is open to the public for demo purposes
# Only limitation of this is that it only returns particular round and season data, to overcome this we can use nested loop through the seasons and rounds to get a full dataset.

# Another viable option is to create an account and then request the administrators to give us the API key for the bulk endpoint, which will allow us to pull all the data we need for our analysis.
def run_live_api_demo():
    print(" [DEMO START] Querying unauthenticated public Jolpica API...")
    
    # Target: 2024 Season, Round 1 (Bahrain Grand Prix)
    # The open endpoints do not require any tokens or authentication!
    season = 2024
    round_num = 1
    

    #use python requests library to send an HTTP GET request to the API endpoint and retrieve the JSON response
    print(f" Fetching live lap times for {season} Round {round_num}...")
    lap_url = f"https://api.jolpi.ca/ergast/f1/{season}/{round_num}/laps.json?limit=1000"
    
    try:
        response = requests.get(lap_url).json()
        race_data = response['MRData']['RaceTable']['Races'][0]
        laps_list = race_data['Laps']
    except Exception as e:
        print(f"Connection error or endpoint limit: {e}")
        return

    # Parsing the JSON structure into a tabular format for analysis
    parsed_laps = []
    for lap in laps_list:
        lap_num = int(lap['number'])
        for timing in lap['Timings']:
            parsed_laps.append({
                'year': season,
                'raceId': race_data['raceName'],
                'driverId': timing['driverId'],
                'LapNumber': lap_num,
                'Position': int(timing['position']),
                'LapTime_Str': timing['time']
            })

    # just a sample demo to see if the logic is correct and if the data is being pulled correctly    
    df = pd.DataFrame(parsed_laps)
    print("Successfully pulled raw lap structures ")

    
    print(" Parsing time strings and engineering features on the fly...")
    
    # Convert text string "1:36.412" into flat float seconds
    def text_time_to_seconds(time_str):
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                return float(parts[0]) * 60 + float(parts[1])
            return float(time_str)
        except:
            return 90.0 # Standard baseline fallback

    df['LapTime_Seconds'] = df['LapTime_Str'].apply(text_time_to_seconds)
    
    # Simulate TyreLife tracking and delta paces per driver group
    df = df.sort_values(by=['driverId', 'LapNumber'])
    df['TyreLife'] = df.groupby('driverId').cumcount() + 1
    df['LapTime_Delta'] = df.groupby('driverId')['LapTime_Seconds'].diff().fillna(0)
    df['Stint'] = 1 # Sample structural filler for the single-race view
    
    # Create predictive target warning rule
    df['ShouldPit'] = np.where(df['LapTime_Delta'] > 1.5, 1, 0)

    print("\n LIVE DEMO SAMPLE (UNAUTHENTICATED ENDPOINT) ")
    sample_cols = ['year', 'raceId', 'driverId', 'LapNumber', 'TyreLife', 'Position', 'LapTime_Seconds', 'LapTime_Delta', 'ShouldPit']
    print(df[sample_cols].head(15).to_string(index=False))
    
    # Extract structural matrix arrays for custom tree fits
    feature_cols = ['LapNumber', 'TyreLife', 'Position', 'Stint', 'LapTime_Seconds', 'LapTime_Delta']
    X = df[feature_cols].to_numpy()
    y = df['ShouldPit'].to_numpy()
    
    print("\n --- MODEL HANDOFF ---")
    print(f"Feature Matrix X Shape: {X.shape} (Rows ready for scratch tree loops)")
    print(f"Target Labels array y Shape: {y.shape}")
    print("\n [SUCCESS] Demo fully operational. Ready to show your supervisor!")

if __name__ == "__main__":
    run_live_api_demo()