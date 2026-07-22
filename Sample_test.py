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
        seasons_df = pd.read_csv(os.path.join(data_dir, "formula_one_season.csv"))
        rounds_df = pd.read_csv(os.path.join(data_dir, "formula_one_round.csv"))
        drivers_df = pd.read_csv(os.path.join(data_dir, "formula_one_driver.csv"))
        laps_df = pd.read_csv(os.path.join(data_dir, "formula_one_lap.csv"))
        
        session_entries_df = pd.read_csv(os.path.join(data_dir, "formula_one_sessionentry.csv"))
        sessions_df = pd.read_csv(os.path.join(data_dir, "formula_one_session.csv"))
        round_entries_df = pd.read_csv(os.path.join(data_dir, "formula_one_roundentry.csv"))
        team_drivers_df = pd.read_csv(os.path.join(data_dir, "formula_one_teamdriver.csv"))

        pit_df = pd.read_csv(os.path.join(data_dir, "formula_one_pitstop.csv"))

    except FileNotFoundError as e:
        print(f"\n [ERROR] Missing CSV files inside '{data_dir}'. Verify folder path.")
        print(f" Details: {e}")
        return

    print(" -> All relational tables loaded successfully.")

    # FILTER & RENAME KEYS FOR RELATIONAL JOINS 
    seasons_era = seasons_df[(seasons_df['year'] >= 2022) & (seasons_df['year'] <= 2025)].copy()
    seasons_era = seasons_era.rename(columns={'id': 'season_id'})

    rounds_era = rounds_df.merge(seasons_era[['season_id', 'year']], on='season_id')
    rounds_era = rounds_era.rename(columns={'id': 'round_id', 'name': 'raceName'})

    pit_prep = pit_df[['lap_id']]
    race_sessions = (sessions_df[sessions_df['type'] == 'R'][['id']].rename(columns={'id': 'session_id'}))
    drivers_prep = drivers_df[['id', 'reference', 'abbreviation']].rename(columns={'id': 'driver_id', 'reference': 'driverCode'})
    team_drivers_prep = team_drivers_df[['id', 'driver_id']].rename(columns={'id': 'team_driver_id'})
    round_entries_prep = round_entries_df[['id', 'round_id', 'team_driver_id']].rename(columns={'id': 'round_entry_id'})
    session_entries_prep = session_entries_df[['id', 'round_entry_id', 'session_id']].rename(columns={'id': 'session_entry_id'}).merge(race_sessions, on='session_id')

    # EXECUTE TABLE MERGES
    td_driver = team_drivers_prep.merge(drivers_prep, on='driver_id')
    re_td = round_entries_prep.merge(td_driver, on='team_driver_id')
    re_round = re_td.merge(rounds_era[['round_id', 'year', 'raceName']], on='round_id')
    se_full = session_entries_prep.merge(re_round, on='round_entry_id')
    df = laps_df.merge(se_full, on='session_entry_id')

    #Additional Pit Column
    df = pd.merge(df, pit_prep, left_on='id', right_on='lap_id', how='left').rename(columns={'lap_id': 'endpoint_shouldpit'})
    #If a timestamp does not exist, then it didn't pit, so it's set to 0.
    df['endpoint_shouldpit'] = df['endpoint_shouldpit'].notna().astype(int)

    df = df.rename(columns={
        'number': 'LapNumber',
        'position': 'Position',
        'time': 'LapTime_Str'
    })

    # TIME CONVERSION UTILITY 
    def text_time_to_seconds(val):
        try:
            if pd.isna(val):
                return 90.0
            val_str = str(val)
            if ':' in val_str:
                parts = val_str.split(':')
                return float(parts[0]) * 60 + float(parts[1])
            return float(val_str)
        except:
            return 90.0

    df['LapTime_Seconds'] = df['LapTime_Str'].apply(text_time_to_seconds)
    
    # Sort chronologically per driver per race
    df = df.sort_values(by=['year', 'raceName', 'driverCode', 'LapNumber']).reset_index(drop=True)

    # FEATURE ENGINEERING & STINT RESETS 
    df['LapTime_Delta'] = df.groupby(['year', 'raceName', 'driverCode'])['LapTime_Seconds'].diff().fillna(0)
    
    # Identify pit stops (lap time spike > 15s indicates pit lane entry)
    df['IsPitLap'] = np.where(df['LapTime_Delta'] > 15.0, 1, 0)
    
    # Increment Stint count and reset TyreLife to 1 on pit stops
    df['Stint'] = df.groupby(['year', 'raceName', 'driverCode'])['IsPitLap'].cumsum() + 1
    df['TyreLife'] = df.groupby(['year', 'raceName', 'driverCode', 'Stint']).cumcount() + 1

    # Target label: ShouldPit = 1 only for organic pace loss (> 1.5s) on track
    df['ShouldPit'] = np.where((df['LapTime_Delta'] > 1.5) & (df['IsPitLap'] == 0), 1, 0)

    #Fixed stint using jolpica pit data
    df['endpoint_Stint'] = df.groupby(['year', 'raceName', 'driverCode'])['endpoint_shouldpit'].transform(lambda x: x.shift(fill_value=0).cumsum() + 1)
    #Fixed TyreLife using jolpica pit data
    df['endpoint_TyreLife'] = df.groupby(['year', 'raceName', 'driverCode', 'endpoint_Stint']).cumcount() + 1

    print(f"\nSuccessfully processed {len(df):,} total laps across 2022-2025.")

    # 70/30 CHRONOLOGICAL TRAIN/TEST SPLIT 
    unique_races = df['raceName'].unique()
    split_boundary = int(len(unique_races) * 0.70)

    train_races = unique_races[:split_boundary]
    test_races = unique_races[split_boundary:]

    feature_cols = ['LapNumber', 'TyreLife', 'Position', 'Stint', 'LapTime_Seconds', 'LapTime_Delta']

    train_df = df[df['raceName'].isin(train_races)]
    test_df = df[df['raceName'].isin(test_races)]

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df['ShouldPit'].to_numpy()

    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df['ShouldPit'].to_numpy()

    print("\n --- MODEL HANDOFF (70/30 CHRONOLOGICAL SPLIT) ---")
    print(f"X_train Matrix Shape: {X_train.shape} | y_train Shape: {y_train.shape}")
    print(f"X_test  Matrix Shape: {X_test.shape}  | y_test  Shape: {y_test.shape}")

    # Export clean master dataset
    csv_name = "f1_lap_data.csv"
    df.to_csv(csv_name, index=False)
    print(f"\n [SUCCESS] Saved master dataset to: {csv_name}")

if __name__ == "__main__":
    process_jolpica_csv_dump()