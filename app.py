import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Set directories and csv locations
BASE_DIR = Path(__file__).resolve().parent
MASTER_CSV = BASE_DIR / "f1_lap_data_master.csv"
MODEL_PATH = BASE_DIR / "final_model.pkl"

MODEL_FEATURES = [
    "LapNumber",
    "is_lap_1",
    "endpoint_TyreLife",
    "Position",
    "endpoint_Stint",
    "LapTime_Seconds",
    "stint_comp"
]

# Load data and model
model = joblib.load(MODEL_PATH)
df = pd.read_csv(MASTER_CSV)

# Set up Streamlit page
st.set_page_config(
    page_title="F1 Pit Stop Optimizer",
    page_icon="🏎️",
    layout="wide"
)

st.title("Formula 1 Pit Stop Optimizer")

# User input selectors
years = sorted(df["year"].unique())
selected_year = st.sidebar.selectbox("Season", years)

race_names = sorted(df[df["year"] == selected_year]["raceName"].unique())
selected_race = st.sidebar.selectbox("Grand Prix", race_names)

race_df = df[
    (df["year"] == selected_year) &
    (df["raceName"] == selected_race)
].copy()

drivers = sorted(race_df["driverCode"].unique())
selected_driver = st.sidebar.selectbox("Driver", drivers)

# Run Model
X = race_df[MODEL_FEATURES]
race_df["Predicted_Pit"] = model.predict(X)

race_df["Actual_Status"] = race_df["endpoint_shouldpit"].map({
    0: "Stay Out",
    1: "Pit Next Lap"
})

race_df["Predicted_Status"] = race_df["Predicted_Pit"].map({
    0: "Stay Out",
    1: "Pit Next Lap"
})

def evaluation(row):
    if row.endpoint_shouldpit == 1 and row.Predicted_Pit == 1:
        return "Correct Pit Stop"
    elif row.endpoint_shouldpit == 0 and row.Predicted_Pit == 1:
        return "False Alarm"
    elif row.endpoint_shouldpit == 1 and row.Predicted_Pit == 0:
        return "Missed Pit Stop"
    else:
        return "Correct Stay Out"

race_df["Evaluation"] = race_df.apply(evaluation, axis=1)
driver_df = race_df[race_df["driverCode"] == selected_driver].copy()

# Summary
st.header(f"{selected_year} {selected_race}")

accuracy = (
    driver_df["endpoint_shouldpit"] ==
    driver_df["Predicted_Pit"]
).mean() * 100

actual_pits = driver_df["endpoint_shouldpit"].sum()

caught_pits = len(
    driver_df[
        (driver_df["endpoint_shouldpit"] == 1) &
        (driver_df["Predicted_Pit"] == 1)
    ]
)

c1, c2, c3 = st.columns(3)

c1.metric("Prediction Accuracy", f"{accuracy:.2f}%")
c2.metric("Actual Pit Stops", int(actual_pits))
c3.metric("Pit Stops Caught", int(caught_pits))

# Driver Table
st.subheader(f"Driver: {selected_driver}")

display_cols = [
    "LapNumber",
    "Position",
    "endpoint_Stint",
    "endpoint_TyreLife",
    "LapTime_Seconds",
    "Actual_Status",
    "Predicted_Status",
    "Evaluation",
]

st.dataframe(
    driver_df[display_cols],
    use_container_width=True,
    hide_index=True
)

# Show Race stats
st.divider()
st.header("Race Statistics")
st.markdown(f"**Race:** {selected_year} {selected_race}")

race_accuracy = (
    race_df["endpoint_shouldpit"] ==
    race_df["Predicted_Pit"]
).mean() * 100

race_actual_pits = int(race_df["endpoint_shouldpit"].sum())
race_predicted_pits = int(race_df["Predicted_Pit"].sum())

race_caught_pits = int(
    (
        (race_df["endpoint_shouldpit"] == 1) &
        (race_df["Predicted_Pit"] == 1)
    ).sum()
)

num_drivers = race_df["driverCode"].nunique()
num_laps = race_df["LapNumber"].max()

col1, col2, col3 = st.columns(3)
col1.metric("Race Accuracy", f"{race_accuracy:.2f}%")
col2.metric("Drivers", num_drivers)
col3.metric("Race Laps", int(num_laps))

col4, col5, col6 = st.columns(3)
col4.metric("Actual Pit Stops", race_actual_pits)
col5.metric("Predicted Pit Stops", race_predicted_pits)
col6.metric("Pit Stops Correctly Identified", race_caught_pits)

# Display model visualizations
st.divider()
st.header("Model Performance & Visualization")

# Decision Tree
with st.expander("Decision Tree Visualization"):
    st.image(
        "decision_tree_diagram.png",
        caption="Decision Tree Classifier",
        use_container_width=True
    )
    
# Confusion Matrix
with st.expander("Confusion Matrix"):
    st.image(
        "confusion_matrix.png",
        caption="Confusion Matrix",
        use_container_width=True
    )

# Group K-Fold Confusion Matrix
with st.expander("Group K-Fold Confusion Matrix"):
    st.image(
        "confusion_matrix_kfold.png",
        caption="Group K-Fold Cross Validation Confusion Matrix",
        use_container_width=True
    )