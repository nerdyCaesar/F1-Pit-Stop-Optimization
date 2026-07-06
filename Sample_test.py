import requests
import pandas as pd
import numpy as np
import time

# This script gets F1 lap timing data from the Jolpica API. The bulk API requires an API key so we used the public endpoint and loop through each season and race to build a complete dataset.
def run_live_api_demo():
    print(" [DEMO START] Querying unauthenticated public Jolpica API...")
    
    all_laps = []
    
    # Download every race from 2022-2025
    for season in range(2022, 2026):
        print(f"Fetching {season} season.")
        schedule_url = f"https://api.jolpi.ca/ergast/f1/{season}.json"

        try:
            schedule = requests.get(schedule_url).json()
            races = schedule['MRData']['RaceTable']['Races']
        except Exception as e:
            print(f"Could not fetch schedule for {season}: {e}")
            continue
        
        print(f"{len(races)} races found.")
        
        # Download lap timing data for every race
        for race in races:
            round_num = race['round']
    
            print(f" Fetching live lap times for {season} Round {round_num}...")
            lap_url = f"https://api.jolpi.ca/ergast/f1/{season}/{round_num}/laps.json?limit=1000"
            
            try:
                response = requests.get(lap_url).json()
                races = response.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if len(races) == 0:
                    print(f"No lap data found for {season} Round {round_num}.")
                    continue
                
                race_data = races[0]
                laps_list = race_data.get("Laps", [])
                
                # Parse the JSON structure into a tabular format for analysis
                for lap in laps_list:
                    lap_num = int(lap['number'])
                    for timing in lap['Timings']:
                        all_laps.append({
                            'year': season,
                            'raceId': race_data['raceName'],
                            'driverId': timing['driverId'],
                            'LapNumber': lap_num,
                            'Position': int(timing['position']),
                            'LapTime_Str': timing['time']
                        })

            except Exception as e:
                print(f"Failed: {season} Round {round_num}. Could not fetch lap data: {e}")
                continue
            
            time.sleep(1)
            
    # Convert collected records into a DataFrame    
    df = pd.DataFrame(all_laps)
    print(f"Successfully pulled raw lap structures. Rows downloaded: {len(df):,}")
    
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
    df = df.sort_values(by=['year', 'raceId', 'driverId', 'LapNumber'])
    df['TyreLife'] = df.groupby(['year', 'raceId', 'driverId']).cumcount() + 1
    df['LapTime_Delta'] = df.groupby(['year', 'raceId', 'driverId'])['LapTime_Seconds'].diff().fillna(0)
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
    csv_name = "f1_lap_data.csv"
    df.to_csv(csv_name, index=False)
    print(f"\n [SUCCESS] CSV File saved successfully as: {csv_name}")

if __name__ == "__main__":
    run_live_api_demo()