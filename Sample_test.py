import requests
import pandas as pd
import numpy as np
import time
import os  
import json  

def run_live_api_demo():
    """
    Main orchestrator function that manages the entire F1 data pipeline:
    1. Fetches historical race schedules and lap timing data from the public Jolpica API.
    2. Implements a local file caching system to prevent redundant API calls (deduplication).
    3. Handles API rate limits with automatic wait-and-retry logic (exponential backoff / 1-hour sleep).
    4. Re-assembles downloaded race JSON files into a single unified Pandas DataFrame.
    5. Performs feature engineering (time conversion, tyre life calculation, and delta timing)
       to produce structured feature matrices (X) and target arrays (y) for a custom decision tree model.
    """
    print(" [DEMO START] Querying unauthenticated public Jolpica API...")
    
    # Stores parsed lap dictionaries before converting them to a Pandas DataFrame
    all_laps = []
    
    # Establish a persistent local folder to cache individual race JSON files.
    # This acts as our local database to avoid hitting the API limit on rerun.
    CACHE_DIR = "f1_race_cache"
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Loop chronologically through the ground-effect regulation era (2022-2025)
    for season in range(2022, 2026):
        print(f"Fetching {season} season.")
        schedule_url = f"https://api.jolpi.ca/ergast/f1/{season}.json"

        try:
            # Query the season schedule to find out how many races occurred and their rounds
            schedule_url_response = requests.get(schedule_url)
            
            # Check if our schedule query is being rate-limited (HTTP status 429).
            # If throttled, sleep for 1 hour to let the unauthenticated API window refresh.
            if schedule_url_response.status_code == 429:
                print("Rate limit hit on schedule endpoint! Sleeping for 1 hour...")
                time.sleep(3600)
                schedule_url_response = requests.get(schedule_url)
                
            schedule = schedule_url_response.json()
            races = schedule['MRData']['RaceTable']['Races']
        except Exception as e:
            print(f"Could not fetch schedule for {season}: {e}")
            continue
        
        print(f"{len(races)} races found.")
        
        # Iterate over every Grand Prix round scheduled in the current season
        for race in races:
            round_num = race['round']
            
            # Local cache verification:
            # Construct a unique file path for this specific race. If we already completed this
            # download in a previous run, skip the network request entirely to save API bandwidth.
            cache_file = os.path.join(CACHE_DIR, f"{season}_round_{round_num}.json")
            if os.path.exists(cache_file):
                print(f" -> [SKIP] {season} Round {round_num} already exists locally.")
                continue
    
            print(f" Fetching live lap times for {season} Round {round_num}...")
            # Request lap data with limit=1000 to fetch as many lap times as possible in a single request
            lap_url = f"https://api.jolpi.ca/ergast/f1/{season}/{round_num}/laps.json?limit=1000"
            
            # Continuous retry loop to guarantee downloading the race even if throttled mid-execution
            while True:
                try:
                    res = requests.get(lap_url)
                    
                    # Intercept API rate limitations (HTTP 429).
                    # Hibernates the script for 1 hour, then loops back to retry the exact same race.
                    if res.status_code == 429:
                        print(f" !!! [429 RATE LIMIT] Hit ceiling at {season} Round {round_num}. Entering 1-hour hibernation...")
                        time.sleep(3600)
                        print(" Waking up. Retrying last request...")
                        continue  
                    
                    response = res.json()
                    
                    # Isolate race records. Uses a safe get() fallback to prevent KeyErrors
                    race_data_list = response.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                    if len(race_data_list) == 0:
                        print(f"No lap data found for {season} Round {round_num}.")
                        break  # Move on if the API simply has no records for this round
                    
                    # Write the raw JSON race dictionary directly to our local cache directory.
                    # This ensures we never have to query this specific race from the web API again.
                    with open(cache_file, 'w') as f:
                        json.dump(race_data_list[0], f)
                    print(f" -> [SAVED] Successfully cached {season} Round {round_num} locally.")
                    break  # Success! Break out of the retry loop to process the next race
                    
                except Exception as e:
                    print(f"Failed: {season} Round {round_num}. Could not fetch lap data: {e}")
                    time.sleep(5) # Brief wait before retrying in case of temporary network dropouts
                    continue
            
            # Polite pause between requests to respect the unauthenticated burst rate limit (4 req/sec)
            time.sleep(0.3)
            
    # Process local files:
    # After checking all seasons, assemble the master dataset from our local JSON cache.
    # This allows the script to run instantly on subsequent executions without hitting the network.
    print("\n[PROCESSING] Building final dataframe from local cache folder...")
    for filename in os.listdir(CACHE_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(CACHE_DIR, filename), 'r') as f:
                race_data = json.load(f)
                
            # Extract year directly from file naming convention (e.g., '2024_round_1.json')
            file_year = int(filename.split('_')[0])
            laps_list = race_data.get("Laps", [])
            
            # Parse nested API JSON into clean, flat relational dictionaries
            for lap in laps_list:
                lap_num = int(lap['number'])
                for timing in lap['Timings']:
                    all_laps.append({
                        'year': file_year,
                        'raceId': race_data['raceName'],
                        'driverId': timing['driverId'],
                        'LapNumber': lap_num,
                        'Position': int(timing['position']),
                        'LapTime_Str': timing['time']
                    })
            
    # Load all gathered dictionaries into a Pandas DataFrame
    df = pd.DataFrame(all_laps)
    
    # Safety Check: Exit gracefully if the script runs but no local files exist to process
    if df.empty:
        print("\n [TERMINATED] No local data found and no new races downloaded. Check your cache folder.")
        return
    
    print(f"Successfully pulled raw lap structures. Rows downloaded: {len(df):,}")
    
    def text_time_to_seconds(time_str):
        """
        Converts F1 timing strings (e.g., "1:36.412" or "90.230") into float seconds.
        If the string is corrupted or missing, defaults to 90.0 seconds to keep arrays numeric.
        """
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                return float(parts[0]) * 60 + float(parts[1])
            return float(time_str)
        except:
            return 90.0 # Standard baseline fallback

    # Apply timing string conversion across the entire dataframe
    df['LapTime_Seconds'] = df['LapTime_Str'].apply(text_time_to_seconds)
    
    # Sort dataset chronologically per driver to calculate sequential metrics
    df = df.sort_values(by=['year', 'raceId', 'driverId', 'LapNumber'])
    
    # Feature Engineering:
    # 1. TyreLife: Calculate cumulative laps run by counting consecutive driver laps.
    # 2. LapTime_Delta: Calculate performance dropoff by finding the time difference from the previous lap.
    #TODO: reset every time the car pits
    df['TyreLife'] = df.groupby(['year', 'raceId', 'driverId']).cumcount() + 1
    df['LapTime_Delta'] = df.groupby(['year', 'raceId', 'driverId'])['LapTime_Seconds'].diff().fillna(0)
    df['Stint'] = 1 # Placeholder structural column to match classification formats
    
    # Define predictive target (ShouldPit):
    # Set target label to '1' (Pit) if a driver drops more than 1.5 seconds off their baseline pace, else '0' (Stay Out).
    df['ShouldPit'] = np.where(df['LapTime_Delta'] > 1.5, 1, 0)

    print("\n LIVE DEMO SAMPLE (UNAUTHENTICATED ENDPOINT) ")
    sample_cols = ['year', 'raceId', 'driverId', 'LapNumber', 'TyreLife', 'Position', 'LapTime_Seconds', 'LapTime_Delta', 'ShouldPit']
    # Sample 15 random entries across the entire multi-season dataset to verify data diversity
    print(df[sample_cols].sample(15).to_string(index=False))
    
    # Model Handoff:
    # Convert numerical features and target labels directly into raw NumPy arrays.
    # This provides the clean matrix input required by custom-built Decision Tree algorithms.
    feature_cols = ['LapNumber', 'TyreLife', 'Position', 'Stint', 'LapTime_Seconds', 'LapTime_Delta']
    X = df[feature_cols].to_numpy()
    y = df['ShouldPit'].to_numpy()
    
    print("\n --- MODEL HANDOFF ---")
    print(f"Feature Matrix X Shape: {X.shape} (Rows ready for scratch tree loops)")
    print(f"Target Labels array y Shape: {y.shape}")
    
    # Export the engineered master dataset to CSV for model reuse
    csv_name = "f1_lap_data.csv"
    df.to_csv(csv_name, index=False)
    print(f"\n [SUCCESS] CSV File saved successfully as: {csv_name}")

if __name__ == "__main__":
    run_live_api_demo()
