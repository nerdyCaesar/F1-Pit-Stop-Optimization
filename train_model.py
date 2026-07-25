import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import data pipeline from Sample_test.py
from Sample_test import process_jolpica_csv_dump

BASE_DIR = Path(__file__).resolve().parent

def run_step_1(X_train, y_train, X_test, y_test, feature_cols):
    """
    Trains the Decision Tree model and saves performance visualisations.
    """
    print("\n [STEP 1] Training Decision Tree Classifier...")

    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"\nOverall Model Accuracy across Test Set: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stay Out (0)', 'Pit Next Lap (1)']))

    # Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Stay Out', 'Pit Next Lap'], 
                yticklabels=['Stay Out', 'Pit Next Lap'])
    plt.title('F1 Pit Strategy - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'confusion_matrix.png')
    plt.close()

    # Save Decision Tree Diagram
    plt.figure(figsize=(18, 9))
    plot_tree(clf, feature_names=feature_cols, class_names=['Stay Out', 'Pit'], filled=True, fontsize=9)
    plt.savefig(BASE_DIR / 'decision_tree_diagram.png', dpi=300)
    plt.close()

    print(" -> Saved 'confusion_matrix.png' and 'decision_tree_diagram.png'.")
    return clf


def demonstrate_race_predictions(clf, df, feature_cols, target_race_name="Abu Dhabi Grand Prix", target_year=2024):
    """
    Evaluates model performance on a single race, displays summary accuracy metrics first,
    and then interactively prompts the user to inspect any driver's specific lap predictions.
    """
    print("\n" + "="*70)
    print(f"      RACE DEMONSTRATION: {target_year} {target_race_name}      ")
    print("="*70)

    # Filter data for chosen race and year
    race_mask = (df['raceName'] == target_race_name) & (df['year'] == target_year)
    race_df = df[race_mask].copy()

    # Fallback to latest available race if target race is missing
    if race_df.empty:
        last_event = df[['year', 'raceName']].drop_duplicates().iloc[-1]
        target_year, target_race_name = last_event['year'], last_event['raceName']
        race_mask = (df['raceName'] == target_race_name) & (df['year'] == target_year)
        race_df = df[race_mask].copy()
        print(f" [INFO] Target race not found. Falling back to: {target_year} {target_race_name}")

    # Generate model predictions for the entire race
    X_race = race_df[feature_cols].to_numpy()
    race_df['Predicted_Pit'] = clf.predict(X_race)

    # Map numerical values to readable labels
    race_df['Actual_Status'] = race_df['endpoint_shouldpit'].map({1: 'PIT NEXT LAP', 0: 'Stay Out'})
    race_df['Predicted_Status'] = race_df['Predicted_Pit'].map({1: 'PIT NEXT LAP', 0: 'Stay Out'})

    def get_match_label(row):
        if row['endpoint_shouldpit'] == 1 and row['Predicted_Pit'] == 1:
            return "MATCH (Correct Pit Call)"
        elif row['endpoint_shouldpit'] == 0 and row['Predicted_Pit'] == 1:
            return "FALSE ALARM (Early Call)"
        elif row['endpoint_shouldpit'] == 1 and row['Predicted_Pit'] == 0:
            return "MISSED PIT STOP"
        return "MATCH (Stay Out)"

    race_df['Evaluation'] = race_df.apply(get_match_label, axis=1)

    display_cols = [
        'driverCode', 'LapNumber', 'endpoint_Stint', 'endpoint_TyreLife', 
        'LapTime_Seconds', 'Actual_Status', 'Predicted_Status', 'Evaluation'
    ]

    # --- 1. PRINT ACCURACY & RACE METRICS FIRST ---
    race_accuracy = (race_df['endpoint_shouldpit'] == race_df['Predicted_Pit']).mean() * 100
    actual_pits = race_df['endpoint_shouldpit'].sum()
    caught_pits = len(race_df[(race_df['endpoint_shouldpit'] == 1) & (race_df['Predicted_Pit'] == 1)])
    
    print("-" * 70)
    print(f" Race Performance Summary for {target_year} {target_race_name}:")
    print(f" - Overall Race Match Accuracy: {race_accuracy:.2f}%")
    print(f" - Pit Stops Caught: {caught_pits} / {actual_pits} actual pit stops")
    print("-" * 70)

    # Export full race predictions to CSV
    output_filename = BASE_DIR / f"race_demo_{target_year}_{target_race_name.replace(' ', '_')}.csv"
    race_df[display_cols].to_csv(output_filename, index=False)
    print(f" -> Exported full race breakdown to '{output_filename}'\n")

    # Get list of unique driver codes present in this race
    available_drivers = sorted(race_df['driverCode'].dropna().unique().tolist())
    print(f"Available Drivers in this race: {', '.join(available_drivers)}")

    # --- 2. INTERACTIVE DRIVER QUERY LOOP ---
    while True:
        user_input = input("\nEnter a Driver Code (e.g., VER, HAM, LEC) or type 'exit' to quit: ").strip().upper()

        if user_input in ['EXIT', 'QUIT', 'Q', '']:
            print("\nExiting demonstration inspection. Done!")
            break

        # Filter telemetry specifically for the chosen driver
        driver_df = race_df[race_df['driverCode'] == user_input]

        if driver_df.empty:
            # Check if input matches partial name if code fails
            matching_rows = race_df[race_df['driverCode'].str.contains(user_input, case=False, na=False)]
            if not matching_rows.empty:
                driver_df = matching_rows
                user_input = driver_df['driverCode'].iloc[0]
            else:
                print(f" [!] Driver '{user_input}' not found in this race. Choose from: {', '.join(available_drivers)}")
                continue

        # Filter driver output to highlight pit stop laps or predicted pit calls
        driver_pit_laps = driver_df[display_cols].sort_values(by='LapNumber')

        print(f"\n" + "="*70)
        print(f" TELEMETRY & PREDICTIONS FOR DRIVER: {user_input}")
        print("="*70)

        if driver_pit_laps.empty:
            print(f"No pit stops or pit triggers recorded for driver {user_input}.")
        else:
            print(driver_pit_laps.to_string(index=False))

        # Show driver-specific accuracy
        driver_acc = (driver_df['endpoint_shouldpit'] == driver_df['Predicted_Pit']).mean() * 100
        driver_actual_pits = driver_df['endpoint_shouldpit'].sum()
        driver_caught_pits = len(driver_df[(driver_df['endpoint_shouldpit'] == 1) & (driver_df['Predicted_Pit'] == 1)])
        print("-" * 70)
        print(f" Driver Summary ({user_input}): Match Accuracy = {driver_acc:.2f}% | Pit Calls Caught = {driver_caught_pits}/{driver_actual_pits}")
        print("-" * 70)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_cols, full_df = process_jolpica_csv_dump()
    clf = run_step_1(X_train, y_train, X_test, y_test, feature_cols)
    demonstrate_race_predictions(clf, full_df, feature_cols, target_race_name="Abu Dhabi Grand Prix", target_year=2024)