import pandas as pd
import numpy as np
import os

def process_jolpica_csv_dump(data_dir="./jolpica-f1-csv"):
    """
    Main pipeline function that loads raw database tables, joins driver and race information,
    engineers features (TyreLife, LapTime_Delta), and prepares 70/30 train/test matrices.
    """
    print(f" [LOCAL PROCESSING] Loading CSV files from '{data_dir}'...")

    # READ RAW DATABASE TABLES FROM DISK 
    try:
        # Load core entity tables
        seasons_df = pd.read_csv(os.path.join(data_dir, "formula_one_season.csv"))
        rounds_df = pd.read_csv(os.path.join(data_dir, "formula_one_round.csv"))
        drivers_df = pd.read_csv(os.path.join(data_dir, "formula_one_driver.csv"))
        laps_df = pd.read_csv(os.path.join(data_dir, "formula_one_lap.csv"))
        
        # Load database junction/mapping tables that connect drivers to sessions and races
        session_entries_df = pd.read_csv(os.path.join(data_dir, "formula_one_sessionentry.csv"))
        round_entries_df = pd.read_csv(os.path.join(data_dir, "formula_one_roundentry.csv"))
        team_drivers_df = pd.read_csv(os.path.join(data_dir, "formula_one_teamdriver.csv"))

    except FileNotFoundError as e:
        print(f"\n [ERROR] Missing CSV files inside '{data_dir}'. Verify folder path.")
        print(f" Details: {e}")
        return

    print(" -> All relational tables loaded successfully.")

    # FILTER & RENAME KEYS FOR RELATIONAL JOINS 
    # Filter only for seasons between 2022 and 2025 (Ground-Effect era)
    seasons_era = seasons_df[(seasons_df['year'] >= 2022) & (seasons_df['year'] <= 2025)].copy()
    seasons_era = seasons_era.rename(columns={'id': 'season_id'})

    # Attach season year to each individual Grand Prix round
    rounds_era = rounds_df.merge(seasons_era[['season_id', 'year']], on='season_id')
    rounds_era = rounds_era.rename(columns={'id': 'round_id', 'name': 'raceName'})

    # Standardize primary key column names across tables to prevent naming conflicts during joins
    drivers_prep = drivers_df[['id', 'reference', 'abbreviation']].rename(columns={'id': 'driver_id', 'reference': 'driverCode'})
    team_drivers_prep = team_drivers_df[['id', 'driver_id']].rename(columns={'id': 'team_driver_id'})
    round_entries_prep = round_entries_df[['id', 'round_id', 'team_driver_id']].rename(columns={'id': 'round_entry_id'})
    session_entries_prep = session_entries_df[['id', 'round_entry_id']].rename(columns={'id': 'session_entry_id'})

    # EXECUTE TABLE MERGES (JOINING THE DATABASE) 
    # Connect team driver entry to driver profile details
    td_driver = team_drivers_prep.merge(drivers_prep, on='driver_id')
    
    # Connect driver profile to round entry details
    re_td = round_entries_prep.merge(td_driver, on='team_driver_id')
    
    # Connect round entry to overall race information (race name, year)
    re_round = re_td.merge(rounds_era[['round_id', 'year', 'raceName']], on='round_id')

    # Connect session entries to race metadata
    se_full = session_entries_prep.merge(re_round, on='round_entry_id')

    # Merge individual lap records with complete driver and race metadata
    df = laps_df.merge(se_full, on='session_entry_id')

    # Standardize column headers for clarity
    df = df.rename(columns={
        'number': 'LapNumber',
        'position': 'Position',
        'time': 'LapTime_Str'
    })

    # TIME CONVERSION UTILITY 
    def text_time_to_seconds(val):
        """Converts strings like '1:36.412' or numbers into total floating-point seconds."""
        try:
            if pd.isna(val):
                return 90.0
            val_str = str(val)
            if ':' in val_str:
                parts = val_str.split(':')
                return float(parts[0]) * 60 + float(parts[1])
            return float(val_str)
        except:
            return 90.0 # Standard fallback if time is missing or unparseable

    # Apply timing conversion across all lap records
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
    sample_cols = ['year', 'raceId', 'driverId', 'LapNumber', 'TyreLife', 'Position', 'LapTime_Seconds', 'LapTime_Delta', 'Base_ShouldPit']
    # Sample 15 random entries across the entire multi-season dataset to verify data diversity
    print(df[sample_cols].sample(15).to_string(index=False))
    
    # Model Handoff:
    # Convert numerical features and target labels directly into raw NumPy arrays.
    # This provides the clean matrix input required by custom-built Decision Tree algorithms.
    feature_cols = ['LapNumber', 'TyreLife', 'Position', 'Stint', 'LapTime_Seconds', 'LapTime_Delta']

    # Filter dataframe into training and testing sets based on race boundary
    train_df = df[df['raceName'].isin(train_races)]
    test_df = df[df['raceName'].isin(test_races)]

    # Convert pandas slices into raw NumPy arrays for high-performance algorithm processing
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df['ShouldPit'].to_numpy()

    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df['ShouldPit'].to_numpy()

    print("\n --- MODEL HANDOFF (70/30 CHRONOLOGICAL SPLIT) ---")
    print(f"X_train Matrix Shape: {X_train.shape} | y_train Shape: {y_train.shape}")
    print(f"X_test  Matrix Shape: {X_test.shape}  | y_test  Shape: {y_test.shape}")

    # Export master dataset to local CSV for easy reuse
    csv_name = "f1_lap_data.csv"
    df.to_csv(csv_name, index=False)
    print(f"\n [SUCCESS] Saved master dataset to: {csv_name}")

if __name__ == "__main__":
    run_live_api_demo()
