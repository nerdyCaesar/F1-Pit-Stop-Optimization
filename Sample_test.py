import pandas as pd
from pathlib import Path

# Set BASE_DIR to script folder
BASE_DIR = Path(__file__).resolve().parent

def process_jolpica_csv_dump(data_dir=None, output_csv=None):
    """
    Reads raw Jolpica F1 relational CSVs, constructs lap-by-lap strategy features,
    calculates shifted targets to avoid data leakage, and prepares feature matrices.
    """
    if data_dir is None:
        data_dir = BASE_DIR / "jolpica-f1-csv"
    if output_csv is None:
        output_csv = BASE_DIR / "f1_lap_data.csv"

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
    seasons_era = seasons_df[(seasons_df['year'] >= 2022) & (seasons_df['year'] <= 2025)].copy()
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

    # Create a unique Group ID for each Grand Prix Event (Year + Round) for GroupKFold
    df['race_group'] = df['year'].astype(str) + "_R" + df['roundNumber'].astype(str)

    print(f"\nSuccessfully processed {len(df):,} total valid laps across 2022-2025.")

    feature_cols = ['LapNumber', 'endpoint_TyreLife', 'Position', 'endpoint_Stint', 'LapTime_Seconds']

    X = df[feature_cols].to_numpy()
    y = df['endpoint_shouldpit'].to_numpy()
    groups = df['race_group'].to_numpy()

    df.to_csv(output_csv, index=False)
    print(f" [SUCCESS] Saved master dataset to: {output_csv}")

    return X, y, groups, feature_cols, df

if __name__ == "__main__":
    process_jolpica_csv_dump()