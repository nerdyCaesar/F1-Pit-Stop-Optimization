# F1 Pit Stop Optimization Model
The F1 Pit Stop Optimizer is a Machine Learning project that predicts whether an Formula 1 driver should make a pit stop on the next lap. The project processes raw Jolpica F1 CSV data from the 2022-2025 seasons and uses a random forest classifier to predict the next pit stop.

The project also includes a Streamlit demo where users can select a season, Grand Prix, and driver to see the model's predictions and compare them with that actually happened.


# Project Overview
Formula 1 is a sport where small decisions can completely change the outcome of a race. One of the most important decisions is knowing the exact right time to make a pit stop. Pitting too early can cause a loss of track position, while pitting too late can lead to tire degradation and slower performance.

The project's research question is:
Based on current race conditions, can a machine learning model accurately predict when an F1 driver should pit?

The model used in this project takes inputs such as:
- Lap number
- Tyre life
- Race Position
- Stint number
- Lap time
- Current stint length compared to typical stints

The model produces a probability of a pit stop.

For example:
Pit Stop Probability: 62%
Threshold: 35%
Prediction: PIT NEXT LAP


# GitHub Structure
```
F1-Pit-Stop-Optimizer/
│
├── jolpica-f1-csv/
│   ├── formula_one_season.csv
│   ├── formula_one_round.csv
│   ├── formula_one_driver.csv
│   ├── formula_one_lap.csv
│   ├── formula_one_sessionentry.csv
│   ├── formula_one_session.csv
│   ├── formula_one_roundentry.csv
│   ├── formula_one_teamdriver.csv
│   └── formula_one_pitstop.csv
│
├── app/
│   └── app.py
│
├── data/
│   ├── f1_lap_data.csv
│   └── f1_lap_data_master.csv
│
├── models/
│   └── final_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── confusion_matrix_kfold.png
│   ├── desicion_tree_diagram.png
│   ├── feature_importance.png
│   ├── feature_importances.png
│   └── model_metrics.json
│
├── src/
│   ├── Sample_test.py
│   └── train_model.py
│
├── .DS_Store
├── .gitignore
├── README.md
└── requirements.txt
```


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

Next, download and place the Jolpica F1 CSV files inside:
`jolpica-f1-csv/`

The required files are:
- formula_one_season.csv
- formula_one_round.csv
- formula_one_driver.csv
- formula_one_lap.csv
- formula_one_sessionentry.csv
- formula_one_session.csv
- formula_one_roundentry.csv
- formula_one_teamdriver.csv
- formula_one_pitstop.csv

Then, process the raw data. Run:

`python sample_test.py`

This creates:
- f1_lap_data_master.csv
- f1_lap_data.csv files

Then, train the model. Run:

`python train_model.py`

This performs the 5-fold Group Cross-Validation and trains the final model. It also creates:
- final_model.pkl
- model_metrics.json
- confusion_matrix_kfold.png
- feature_importance.png

Finally, launch the Streamlit application. Run:

`streamlit run app.py`

The Streamlit dashboard should then open in your browser.
