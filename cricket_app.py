import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(page_title="Cricket Win Probability Predictor")

st.title("🏏 Cricket Match Win Probability Predictor")
st.write("Prediction is valid after **5 overs**")

# Load trained model
with open("win_model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------- User Inputs ----------

current_score = st.number_input("Current Score", min_value=0)

overs = st.number_input(
    "Overs Completed",
    min_value=5,
    max_value=20,
    step=1
)

balls = st.number_input(
    "Balls in Current Over",
    min_value=0,
    max_value=5,
    step=1
)

# Internal ML value
overs_completed = overs + balls / 10

# Cricket-style display
overs_display = f"{overs}.{balls}"
st.write(f"Overs Completed (cricket format): **{overs_display}**")

wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10)
target_score = st.number_input("Target Score", min_value=0)

# ---------- Match Calculations ----------

TOTAL_OVERS = 20

balls_bowled = overs * 6 + balls
total_balls = TOTAL_OVERS * 6
balls_remaining = total_balls - balls_bowled

runs_needed = target_score - current_score

# Current Run Rate (CRR)
if overs_completed > 0:
    current_run_rate = current_score / overs_completed
else:
    current_run_rate = 0

# Required Run Rate (RRR)
remaining_overs = TOTAL_OVERS - overs_completed
if remaining_overs > 0:
    required_run_rate = runs_needed / remaining_overs
else:
    required_run_rate = 0

# ---------- Display Match Situation ----------

st.subheader("📊 Match Situation")

st.write(f"**Runs Needed:** {runs_needed}")
st.write(f"**Balls Remaining:** {balls_remaining}")
st.write(f"**Current Run Rate (CRR):** {current_run_rate:.2f}")
st.write(f"**Required Run Rate (RRR):** {required_run_rate:.2f}")

# ---------- Prediction ----------

if st.button("Predict Win Probability"):
    input_df = pd.DataFrame([{
        "current_score": current_score,
        "overs_completed": overs_completed,
        "wickets_lost": wickets_lost,
        "target_score": target_score,
        "required_run_rate": required_run_rate
    }])

    probability = model.predict_proba(input_df)[0][1] * 100

    st.success(f"Winning Probability: {probability:.2f}%")
