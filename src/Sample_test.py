import pandas as pd
from pathlib import Path

# Set directories and csv locations
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "jolpica-f1-csv"
MASTER_CSV = BASE_DIR / "data" / "f1_lap_data_master.csv"
TRAINING_CSV = BASE_DIR / "data" / "f1_lap_data.csv"

MODEL_FEATURES = [
    "LapNumber",
    "is_lap_1",
    "endpoint_TyreLife",
    "Position",
    "endpoint_Stint",
    "LapTime_Seconds",
    "stint_comp"
]

START_YEAR = 2022
END_YEAR = 2025
GLOBAL_STINT_LENGTH = 18 # Arbitrary Stint Length var
    
def process_jolpica_csv_dump(data_dir=None, master_csv=None, training_csv=None):
    """
    Reads raw Jolpica F1 relational CSVs, constructs lap-by-lap strategy features,
    calculates shifted targets to avoid data leakage, and prepares feature matrices.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if master_csv is None:
        master_csv = MASTER_CSV
    if training_csv is None:
        training_csv = TRAINING_CSV
    
    print(f" [LOCAL PROCESSING] Loading CSV files from '{data_dir}'...")

    try:
        seasons_df = pd.read_csv(data_dir / "formula_one_season.csv")
        rounds_df = pd.read_csv(data_dir / "formula_one_round.csv")
        drivers_df = pd.read_csv(data_dir / "formula_one_driver.csv")
        laps_df = pd.read_csv(data_dir / "formula_one_lap.csv")
        
        session_entries_df = pd.read_csv(data_dir / "formula_one_sessionentry.csv")
        sessions_df = pd.read_csv(data_dir / "formula_one_session.csv")
        round_entries_df = pd.read_csv(data_dir / "formula_one_roundentry.csv")
        team_drivers_df = pd.read_csv(data_dir / "formula_one_teamdriver.csv")
        pit_df = pd.read_csv(data_dir / "formula_one_pitstop.csv")

    except FileNotFoundError as e:
        print(f"\n [CRITICAL ERROR] Missing required CSV file in '{data_dir}'.")
        raise FileNotFoundError(f"Pipeline stopped due to missing file in '{data_dir}'.") from e

    print(" -> All relational tables loaded successfully.")

    # Filter for modern era seasons (2022-2025)
    seasons_era = seasons_df[(seasons_df['year'] >= START_YEAR) & (seasons_df['year'] <= END_YEAR)].copy()
    seasons_era = seasons_era.rename(columns={'id': 'season_id'})

    rounds_era = rounds_df.merge(seasons_era[['season_id', 'year']], on='season_id')
    rounds_era = rounds_era.rename(columns={'id': 'round_id', 'name': 'raceName', 'number': 'roundNumber'})

    # RelationalJoins
    race_sessions = sessions_df[sessions_df['type'] == 'R'][['id']].rename(columns={'id': 'session_id'})
    drivers_prep = drivers_df[['id', 'reference', 'abbreviation']].rename(columns={'id': 'driver_id', 'reference': 'driverCode'})
    team_drivers_prep = team_drivers_df[['id', 'driver_id']].rename(columns={'id': 'team_driver_id'})
    round_entries_prep = round_entries_df[['id', 'round_id', 'team_driver_id']].rename(columns={'id': 'round_entry_id'})
    session_entries_prep = session_entries_df[['id', 'round_entry_id', 'session_id']].rename(columns={'id': 'session_entry_id'}).merge(race_sessions, on='session_id')

    td_driver = team_drivers_prep.merge(drivers_prep, on='driver_id')
    re_td = round_entries_prep.merge(td_driver, on='team_driver_id')
    re_round = re_td.merge(rounds_era[['round_id', 'year', 'roundNumber', 'raceName']], on='round_id')
    se_full = session_entries_prep.merge(re_round, on='round_entry_id')
    df = laps_df.merge(se_full, on='session_entry_id')

    # Pit stop flags
    df = df.merge(pit_df[['lap_id']], left_on='id', right_on='lap_id', how='left').rename(columns={'lap_id': 'endpoint_shouldpit'})

    df = df.rename(columns={
        'number': 'LapNumber',
        'position': 'Position',
        'time': 'LapTime_Str'
    })

    # Efficient Time Parsing via pd.to_timedelta
    df['LapTime_Seconds'] = pd.to_timedelta(df['LapTime_Str'], errors='coerce').dt.total_seconds()
    df = df.dropna(subset=['LapTime_Seconds']).copy()

    # Chronological sort
    df = df.sort_values(by=['year', 'roundNumber', 'driverCode', 'LapNumber']).reset_index(drop=True)

    # Shift pit target back by 1 lap: telemetry on Lap N predicts if driver pits on Lap N+1
    df['endpoint_shouldpit'] = (
        df['endpoint_shouldpit']
        .notna()
        .groupby([df['year'], df['roundNumber'], df['driverCode']])
        .shift(-1)
        .fillna(0)
        .astype(int)
    )

    # Calculate Stint and Tyre Life vectorially
    df['endpoint_Stint'] = df.groupby(['year', 'roundNumber', 'driverCode'])['endpoint_shouldpit'].transform(
        lambda x: x.shift(2, fill_value=0).cumsum() + 1
    )
    df['endpoint_TyreLife'] = df.groupby(['year', 'roundNumber', 'driverCode', 'endpoint_Stint']).cumcount() + 1

    # Store the max stint of each [raceName, year, driver] 
    df['max_stint'] = df.groupby(['year', 'raceName', 'driverCode'])['endpoint_Stint'].transform('max')
    
    #Calculate length of each stint as a new datafram
    stint_table = (df.groupby(['year', 'raceName', 'driverCode', 'endpoint_Stint']).aggregate(stint_length = ('LapNumber', 'size'), max_stint = ('max_stint', 'first')))

    #reset the index of the table
    stint_table = stint_table.reset_index()

    # Filter out the max stints cause they don't end in a pit
    valid_stints_df = stint_table[stint_table['endpoint_Stint'] != stint_table['max_stint']]
    
    # Calculate median per race and year
    race_medians = valid_stints_df.groupby(['raceName', 'year'])['stint_length'].median().reset_index()
    
    #create prior length column that calculates race medians only for prior years
    race_medians = race_medians.sort_values(['raceName', 'year'])
    race_medians['prior_length'] = race_medians.groupby('raceName')['stint_length'].transform(lambda s: s.shift().expanding().median())

    #Fill in prior_length var for earliest year with arbitrary value
    race_medians['prior_length'] = race_medians['prior_length'].fillna(GLOBAL_STINT_LENGTH)

    # Map back each race median to the orignal df and create current v. typical stint ratio
    df = df.merge(race_medians[['raceName', 'year', 'prior_length']], on=['raceName', 'year'], how='left')
    df['stint_comp'] = df['endpoint_TyreLife'] / df['prior_length']

    # Drop helper columns
    df = df.drop(columns=['max_stint', 'prior_length'])

    # Feature Engineering: Add binary is_lap_1 feature
    df['is_lap_1'] = (df['LapNumber'] == 1).astype(int)

    # Create a unique Group ID for each Grand Prix Event (Year + Round) for GroupKFold
    df['race_group'] = df['year'].astype(str) + "_R" + df['roundNumber'].astype(str)

    print(f"\nSuccessfully processed {len(df):,} total valid laps across 2022-2025.")

    # Rename columns for master CSV
    column_order = [
        # Race metadata
        "year",
        "roundNumber",
        "race_group",
        "raceName",
        "driverCode",
        
        # Lap information
        "LapNumber",
        "Position",
        "LapTime_Str",
        "LapTime_Seconds",
    
        # Engineered features
        "is_lap_1",
        "endpoint_Stint",
        "endpoint_TyreLife",
        "stint_comp",
    
        # Target
        "endpoint_shouldpit",
    ]
    
    remaining_columns = [
        c for c in df.columns
        if c not in column_order
    ]
    
    df = df[column_order + remaining_columns]
    
    # Save master csv
    df.to_csv(master_csv, index=False)
    print(f" [SUCCESS] Saved master dataset to: {master_csv}")

    # Create training dataset
    training_df = df[MODEL_FEATURES + ['endpoint_shouldpit', 'race_group']].copy()
    
    training_df.to_csv(training_csv, index=False)
    print(f" [SUCCESS] Saved training dataset to: {training_csv}")

    X = training_df[MODEL_FEATURES].to_numpy()
    y = training_df['endpoint_shouldpit'].to_numpy()
    groups = training_df['race_group'].to_numpy()
    
    master_df = df
    return (X, y, groups, MODEL_FEATURES, master_df)

if __name__ == "__main__":
    process_jolpica_csv_dump()