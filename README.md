# F1 Pit Stop Optimization Model
The F1 Pit Stop Optimizer is a Machine Learning project that predicts whether an Formula 1 driver should make a pit stop on the next lap. The project processes raw Jolpica F1 CSV data from the 2022-2025 seasons and uses a random forest classifier to predict the next pit stop.

The project also includes a Streamlit demo where users can select a season, Grand Prix, and driver to see the model's predictions and compare them with that actually happened.


# File Directory

The project has three main Python files.

1. `sample_test.py` -> Prepares the raw F1 data.

    - Combines the different F1 data tables.
    - Filters data to the 2022–2025 seasons.
    - Creates lap, tire, stint, and pit stop features.
    - Creates the training dataset.

2. `train_model.py` -> Trains and tests the machine learning model.

    - Uses Group K-Fold Cross-Validation.
    - Uses a custom 35% probability threshold.
    - Measures accuracy, precision, recall, F1-score, and PR-AUC.
    - Saves the trained model and evaluation results.
    - Creates confusion matrix and visualizations.

3. `app.py` -> Creates the interactive Streamlit website.

    - Allows the user to select a season, Grand Prix, and driver.
    - Displays lap by lap pit stop predictions.
    - Compares predictions with actual pit stops.
    - Displays driver and race statistics.

Additionally, there are two different CSV files.

1. `f1_lap_data_master.csv` -> Contains the complete processed dataset.

2. `f1_lap_data.csv` -> Contains only model-related columns.


# Running the Project

First, install the dependencies. Run:

`pip install numpy pandas matplotlib seaborn scikit-learn joblib streamlit`

Then, process the raw data. Run:

`python sample_test.py`

This creates:
- f1_lap_data_master.csv
- f1_lap_data.csv files

Then, train the model. Run:

`python train_model.py`

This performs the 5-fold Group Cross-Validation and trains the final model.

It also creates:
- final_model.pkl
- model_metrics.json
- confusion_matrix_kfold.png
- feature_importance.png

Finally, launch the Streamlit application. Run:

`streamlit run app.py`