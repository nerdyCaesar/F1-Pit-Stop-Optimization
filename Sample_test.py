import pandas as pd
import numpy as np
import os
from pathlib import Path

# Set BASE_DIR to the folder where this script lives, ensuring file paths work anywhere
BASE_DIR = Path(__file__).resolve().parent

def process_jolpica_csv_dump(data_dir=None, output_csv=None):
    """
    This is the main data pipeline function.
    It reads raw Formula 1 database files, links them together, cleans up bad lap times,
    creates custom features like tyre age, sets up our target variable, and splits the data
    chronologically into training and testing sets.
    """
    
    # Use default folder locations if custom paths aren't provided
    if data_dir is None:
        data_dir = BASE_DIR / "jolpica-f1-csv"
    if output_csv is None:
        output_csv = BASE_DIR / "f1_lap_data.csv"

    print(f" [LOCAL PROCESSING] Loading CSV files from '{data_dir}'...")

    # Read all the individual relational database tables from disk
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
        print(f"\n [CRITICAL ERROR] Missing required CSV file in '{data_dir}'.")
        raise FileNotFoundError(f"Pipeline stopped due to missing file in '{data_dir}'.") from e

    print(" -> All relational tables loaded successfully.")

    # Filter for modern era seasons (2022-2025) and clean up column names
    seasons_era = seasons_df[(seasons_df['year'] >= 2022) & (seasons_df['year'] <= 2025)].copy()
    seasons_era = seasons_era.rename(columns={'id': 'season_id'})

    rounds_era = rounds_df.merge(seasons_era[['season_id', 'year']], on='season_id')
    rounds_era = rounds_era.rename(columns={'id': 'round_id', 'name': 'raceName', 'number': 'roundNumber'})

    # Extract key ID columns from supporting tables so we can join them together
    pit_prep = pit_df[['lap_id']]
    race_sessions = sessions_df[sessions_df['type'] == 'R'][['id']].rename(columns={'id': 'session_id'})
    drivers_prep = drivers_df[['id', 'reference', 'abbreviation']].rename(columns={'id': 'driver_id', 'reference': 'driverCode'})
    team_drivers_prep = team_drivers_df[['id', 'driver_id']].rename(columns={'id': 'team_driver_id'})
    round_entries_prep = round_entries_df[['id', 'round_id', 'team_driver_id']].rename(columns={'id': 'round_entry_id'})
    session_entries_prep = session_entries_df[['id', 'round_entry_id', 'session_id']].rename(columns={'id': 'session_entry_id'}).merge(race_sessions, on='session_id')

    # Join all relational tables into one big DataFrame containing every lap
    td_driver = team_drivers_prep.merge(drivers_prep, on='driver_id')
    re_td = round_entries_prep.merge(td_driver, on='team_driver_id')
    re_round = re_td.merge(rounds_era[['round_id', 'year', 'roundNumber', 'raceName']], on='round_id')
    se_full = session_entries_prep.merge(re_round, on='round_entry_id')
    df = laps_df.merge(se_full, on='session_entry_id')

    # Match pit stop events directly to the specific lap IDs where they occurred
    df = pd.merge(df, pit_prep, left_on='id', right_on='lap_id', how='left').rename(columns={'lap_id': 'endpoint_shouldpit'})

    # Rename key columns to readable names
    df = df.rename(columns={
        'number': 'LapNumber',
        'position': 'Position',
        'time': 'LapTime_Str'
    })

    # Parse lap time text (e.g. "1:28.412") into numeric seconds (88.412 seconds)
    df['LapTime_Seconds'] = pd.to_timedelta(df['LapTime_Str'], errors='coerce').dt.total_seconds()
    
    # Drop any corrupted or missing lap time entries to keep data pure
    df = df.dropna(subset=['LapTime_Seconds']).copy()

    # Create a unique key for each race event using (year, roundNumber) to keep exact order
    df['event_key'] = list(zip(df['year'], df['roundNumber']))
    
    # Sort data chronologically per driver per race
    df = df.sort_values(by=['year', 'roundNumber', 'driverCode', 'LapNumber']).reset_index(drop=True)

    # Target Shift (Prevents Data Leakage)
    # Shift the pit stop flag back by 1 lap so telemetry on Lap N predicts if the driver PITS ON LAP N+1
    df['endpoint_shouldpit'] = (
        df['endpoint_shouldpit']
        .notna()
        .groupby([df['year'], df['roundNumber'], df['driverCode']])
        .shift(-1)
        .fillna(0)
        .astype(int)
    )

    # Feature Engineering
    # Calculate current stint number (increments whenever a pit stop occurs)
    df['endpoint_Stint'] = df.groupby(['year', 'roundNumber', 'driverCode'])['endpoint_shouldpit'].transform(
        lambda x: x.shift(2, fill_value=0).cumsum() + 1
    )
    # Track age of tyres (counts laps completed on the current set, resets to 1 on a new stint)
    df['endpoint_TyreLife'] = df.groupby(['year', 'roundNumber', 'driverCode', 'endpoint_Stint']).cumcount() + 1

    print(f"\nSuccessfully processed {len(df):,} total valid laps across 2022-2025.")

    # True Chronological Train/Test Split
    # Sort all distinct Grand Prix events chronologically
    unique_events = df[['year', 'roundNumber']].drop_duplicates().sort_values(by=['year', 'roundNumber'])
    
    # Use the first 70% of historical races for training, and the remaining 30% for testing
    split_boundary = int(len(unique_events) * 0.70)
    train_events = set(map(tuple, unique_events.iloc[:split_boundary].to_numpy()))
    
    train_mask = df['event_key'].isin(train_events)
    train_df = df[train_mask]
    test_df = df[~train_mask]

    # Features used by the Decision Tree model to make decisions
    feature_cols = ['LapNumber', 'endpoint_TyreLife', 'Position', 'endpoint_Stint', 'LapTime_Seconds']

    # Convert pandas columns into raw NumPy matrices for scikit-learn
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df['endpoint_shouldpit'].to_numpy()

    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df['endpoint_shouldpit'].to_numpy()

    print("\n --- MODEL HANDOFF (70/30 CHRONOLOGICAL SPLIT) ---")
    print(f"X_train Matrix Shape: {X_train.shape} | y_train Shape: {y_train.shape}")
    print(f"X_test  Matrix Shape: {X_test.shape}  | y_test  Shape: {y_test.shape}")

    # Export clean combined dataset to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n [SUCCESS] Saved master dataset to: {output_csv}")

    # Return all 6 objects needed by train_model.py
    return X_train, y_train, X_test, y_test, feature_cols, df

if __name__ == "__main__":
    process_jolpica_csv_dump()