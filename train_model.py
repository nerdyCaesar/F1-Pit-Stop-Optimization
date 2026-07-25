import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import the data pipeline function from Sample_test.py
from Sample_test import process_jolpica_csv_dump

# Define base directory path relative to this script
BASE_DIR = Path(__file__).resolve().parent

def run_step_1(X_train, y_train, X_test, y_test, feature_cols):
    """
    Trains a Scikit-Learn Decision Tree model on the training set,
    evaluates its performance on the unseen test set, and saves visual charts.
    """
    print("\n [STEP 1] Training Decision Tree Classifier...")

    # Initialize Decision Tree model:
    # - max_depth=5 prevents the tree from becoming too deep/overfitting
    # - class_weight='balanced' forces the model to pay extra attention to rare pit stop laps (~5% of data)
    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42)
    
    # Train the tree using the training data
    clf.fit(X_train, y_train)

    # Make pit stop predictions on unseen test races
    y_pred = clf.predict(X_test)

    # Output overall performance metrics to console
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stay Out (0)', 'Pit Next Lap (1)']))

    # Generate and save Confusion Matrix plot (heatmap of correct vs incorrect calls)
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

    # Generate and save visual Decision Tree Diagram showing all logical rules
    plt.figure(figsize=(18, 9))
    plot_tree(clf, feature_names=feature_cols, class_names=['Stay Out', 'Pit'], filled=True, fontsize=9)
    plt.savefig(BASE_DIR / 'decision_tree_diagram.png', dpi=300)
    plt.close()

    print(" -> Saved 'confusion_matrix.png' and 'decision_tree_diagram.png'.")
    return clf


def demonstrate_race_predictions(clf, df, feature_cols, target_race_name="Abu Dhabi Grand Prix", target_year=2024):
    """
    Filters the dataset down to a single Grand Prix event and runs predictions lap-by-lap.
    Prints a clear table comparing actual pit stops against model pit calls.
    """
    print("\n" + "="*70)
    print(f"      RACE DEMONSTRATION: {target_year} {target_race_name}      ")
    print("="*70)

    # Filter data for the chosen race and year
    race_mask = (df['raceName'] == target_race_name) & (df['year'] == target_year)
    race_df = df[race_mask].copy()

    # If the requested race isn't in the dataset, fall back to the very last available race
    if race_df.empty:
        last_event = df[['year', 'raceName']].drop_duplicates().iloc[-1]
        target_year, target_race_name = last_event['year'], last_event['raceName']
        race_mask = (df['raceName'] == target_race_name) & (df['year'] == target_year)
        race_df = df[race_mask].copy()
        print(f" [INFO] Target race not found. Falling back to: {target_year} {target_race_name}")

    # Extract feature matrix for this specific race and predict pit decisions
    X_race = race_df[feature_cols].to_numpy()
    race_df['Predicted_Pit'] = clf.predict(X_race)

    # Convert binary 0 and 1 values into readable text labels
    race_df['Actual_Status'] = race_df['endpoint_shouldpit'].map({1: 'PIT NEXT LAP', 0: 'Stay Out'})
    race_df['Predicted_Status'] = race_df['Predicted_Pit'].map({1: 'PIT NEXT LAP', 0: 'Stay Out'})

    # Compare actual vs predicted decisions to evaluate prediction quality
    def get_match_label(row):
        if row['endpoint_shouldpit'] == 1 and row['Predicted_Pit'] == 1:
            return "MATCH (Correct Pit Call)"
        elif row['endpoint_shouldpit'] == 0 and row['Predicted_Pit'] == 1:
            return "FALSE ALARM (Early Call)"
        elif row['endpoint_shouldpit'] == 1 and row['Predicted_Pit'] == 0:
            return "MISSED PIT STOP"
        return "MATCH (Stay Out)"

    race_df['Evaluation'] = race_df.apply(get_match_label, axis=1)

    # Select key telemetry columns to display in the output table
    display_cols = [
        'driverCode', 'LapNumber', 'endpoint_Stint', 'endpoint_TyreLife', 
        'LapTime_Seconds', 'Actual_Status', 'Predicted_Status', 'Evaluation'
    ]
    
    # Filter view to focus on laps where a pit stop was either planned or predicted
    pit_laps_and_triggers = race_df[
        (race_df['endpoint_shouldpit'] == 1) | (race_df['Predicted_Pit'] == 1)
    ][display_cols].sort_values(by=['driverCode', 'LapNumber'])

    print(f"\nShowing Pit Window Laps for {target_year} {target_race_name}:")
    print(pit_laps_and_triggers.head(20).to_string(index=False))

    # Calculate race performance statistics
    race_accuracy = (race_df['endpoint_shouldpit'] == race_df['Predicted_Pit']).mean() * 100
    actual_pits = race_df['endpoint_shouldpit'].sum()
    caught_pits = len(race_df[(race_df['endpoint_shouldpit'] == 1) & (race_df['Predicted_Pit'] == 1)])
    
    print("-" * 70)
    print(f" Race Performance Summary:")
    print(f" - Overall Race Match Accuracy: {race_accuracy:.2f}%")
    print(f" - Pit Stops Caught: {caught_pits} / {actual_pits} actual pit stops")
    print("-" * 70)

    # Save detailed race breakdown to CSV file
    output_filename = BASE_DIR / f"race_demo_{target_year}_{target_race_name.replace(' ', '_')}.csv"
    race_df[display_cols].to_csv(output_filename, index=False)
    print(f" -> Exported full race breakdown to '{output_filename}'")


# Main execution block
if __name__ == "__main__":
    # Run data processing pipeline to generate matrices and dataset
    X_train, y_train, X_test, y_test, feature_cols, full_df = process_jolpica_csv_dump()
    
    # Train Decision Tree model and export diagram/confusion matrix
    clf = run_step_1(X_train, y_train, X_test, y_test, feature_cols)
    
    # Run lap-by-lap race prediction demonstration
    demonstrate_race_predictions(clf, full_df, feature_cols, target_race_name="Abu Dhabi Grand Prix", target_year=2024)