import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from Sample_test import process_jolpica_csv_dump

BASE_DIR = Path(__file__).resolve().parent

def run_step_1_kfold(X, y, groups, feature_cols, n_splits=5):
    """
    Executes Group K-Fold Cross-Validation across distinct Grand Prix events.
    Trains a final production model on the full dataset once evaluation finishes.
    """
    print(f"\n [STEP 1] Running {n_splits}-Fold Group Cross-Validation (Grouped by Grand Prix)...")

    gkf = GroupKFold(n_splits=n_splits)
    fold_accuracies, fold_precisions, fold_recalls, fold_f1s = [], [], [], []
    
    oof_predictions = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        clf = DecisionTreeClassifier(
            max_depth=5, 
            min_samples_leaf=10, 
            class_weight={0: 1, 1: 8}, # Rebalanced weight to control trigger sensitivity
            random_state=42
        )
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_va)

        oof_predictions[val_idx] = preds

        acc = accuracy_score(y_va, preds)
        prec = precision_score(y_va, preds, pos_label=1, zero_division=0)
        rec = recall_score(y_va, preds, pos_label=1, zero_division=0)
        f1 = f1_score(y_va, preds, pos_label=1, zero_division=0)

        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)

        print(f" Fold {fold} | Acc: {acc*100:.2f}% | Pit Precision: {prec*100:.2f}% | Pit Recall: {rec*100:.2f}% | Pit F1: {f1:.2f}")

    print("\n" + "="*60)
    print(" K-FOLD CROSS-VALIDATION SUMMARY RESULTS")
    print("="*60)
    print(f" Mean Overall Accuracy : {np.mean(fold_accuracies)*100:.2f}% (+/- {np.std(fold_accuracies)*100:.2f}%)")
    print(f" Mean Pit Precision    : {np.mean(fold_precisions)*100:.2f}%")
    print(f" Mean Pit Recall       : {np.mean(fold_recalls)*100:.2f}%")
    print(f" Mean Pit F1-Score     : {np.mean(fold_f1s):.2f}")
    print("="*60)

    print("\nOut-of-Fold Classification Report:")
    print(classification_report(y, oof_predictions, target_names=['Stay Out (0)', 'Pit Next Lap (1)']))

    # Save OOF Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y, oof_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Stay Out', 'Pit Next Lap'], 
                yticklabels=['Stay Out', 'Pit Next Lap'])
    plt.title('Out-of-Fold Confusion Matrix (GroupKFold)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'confusion_matrix_kfold.png')
    plt.close()

    # Train final deployment model on 100% of the dataset
    print("\n -> Training final production Decision Tree model on full dataset...")
    final_clf = DecisionTreeClassifier(
        max_depth=5, 
        min_samples_leaf=10, 
        class_weight={0: 1, 1: 8}, 
        random_state=42
    )
    final_clf.fit(X, y)

    # Save Decision Tree Diagram
    plt.figure(figsize=(18, 9))
    plot_tree(final_clf, feature_names=feature_cols, class_names=['Stay Out', 'Pit'], filled=True, fontsize=9)
    plt.savefig(BASE_DIR / 'decision_tree_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(" -> Saved 'confusion_matrix_kfold.png' and 'decision_tree_diagram.png'.")
    return final_clf


def demonstrate_race_predictions(clf, df, feature_cols, target_race_name="Abu Dhabi Grand Prix", target_year=2024):
    """
    Evaluates model performance on a single target race and interactively queries driver telemetry.
    """
    print("\n" + "="*70)
    print(f"     RACE DEMONSTRATION: {target_year} {target_race_name}      ")
    print("="*70)

    race_mask = (df['raceName'] == target_race_name) & (df['year'] == target_year)
    race_df = df[race_mask].copy()

    if race_df.empty:
        last_event = df[['year', 'raceName']].drop_duplicates().iloc[-1]
        target_year, target_race_name = last_event['year'], last_event['raceName']
        race_mask = (df['raceName'] == target_race_name) & (df['year'] == target_year)
        race_df = df[race_mask].copy()
        print(f" [INFO] Target race not found. Falling back to: {target_year} {target_race_name}")

    X_race = race_df[feature_cols].to_numpy()
    race_df['Predicted_Pit'] = clf.predict(X_race)

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
        'driverCode', 'LapNumber', 'is_lap_1', 'endpoint_Stint', 'endpoint_TyreLife', 
        'LapTime_Seconds', 'Actual_Status', 'Predicted_Status', 'Evaluation'
    ]

    race_accuracy = (race_df['endpoint_shouldpit'] == race_df['Predicted_Pit']).mean() * 100
    actual_pits = race_df['endpoint_shouldpit'].sum()
    caught_pits = len(race_df[(race_df['endpoint_shouldpit'] == 1) & (race_df['Predicted_Pit'] == 1)])
    
    print("-" * 70)
    print(f" Race Performance Summary for {target_year} {target_race_name}:")
    print(f" - Overall Race Match Accuracy: {race_accuracy:.2f}%")
    print(f" - Pit Stops Caught: {caught_pits} / {actual_pits} actual pit stops")
    print("-" * 70)

    output_filename = BASE_DIR / f"race_demo_{target_year}_{target_race_name.replace(' ', '_')}.csv"
    race_df[display_cols].to_csv(output_filename, index=False)
    print(f" -> Exported full race breakdown to '{output_filename}'\n")

    available_drivers = sorted(race_df['driverCode'].dropna().unique().tolist())
    print(f"Available Drivers in this race: {', '.join(available_drivers)}")

    while True:
        user_input = input("\nEnter a Driver Code (e.g., VER, HAM, LEC) or type 'exit' to quit: ").strip().upper()

        if user_input in ['EXIT', 'QUIT', 'Q', '']:
            print("\nExiting demonstration inspection. Done!")
            break

        driver_df = race_df[race_df['driverCode'] == user_input]

        if driver_df.empty:
            matching_rows = race_df[race_df['driverCode'].str.contains(user_input, case=False, na=False)]
            if not matching_rows.empty:
                driver_df = matching_rows
                user_input = driver_df['driverCode'].iloc[0]
            else:
                print(f" [!] Driver '{user_input}' not found in this race. Choose from: {', '.join(available_drivers)}")
                continue

        driver_pit_laps = driver_df[display_cols].sort_values(by='LapNumber')

        print(f"\n" + "="*70)
        print(f" TELEMETRY & PREDICTIONS FOR DRIVER: {user_input}")
        print("="*70)

        if driver_pit_laps.empty:
            print(f"No telemetry records found for driver {user_input}.")
        else:
            print(driver_pit_laps.to_string(index=False))

        driver_acc = (driver_df['endpoint_shouldpit'] == driver_df['Predicted_Pit']).mean() * 100
        driver_actual_pits = driver_df['endpoint_shouldpit'].sum()
        driver_caught_pits = len(driver_df[(driver_df['endpoint_shouldpit'] == 1) & (driver_df['Predicted_Pit'] == 1)])
        print("-" * 70)
        print(f" Driver Summary ({user_input}): Match Accuracy = {driver_acc:.2f}% | Pit Calls Caught = {driver_caught_pits}/{driver_actual_pits}")
        print("-" * 70)


if __name__ == "__main__":
    X, y, groups, feature_cols, full_df = process_jolpica_csv_dump()
    clf = run_step_1_kfold(X, y, groups, feature_cols, n_splits=5)
    demonstrate_race_predictions(clf, full_df, feature_cols, target_race_name="Abu Dhabi Grand Prix", target_year=2024)