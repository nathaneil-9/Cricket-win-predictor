import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("dataset.csv")

# 2. Use only 2nd innings
df = df[df["innings"] == 2].copy()

# 3. Runs per ball
df["runs"] = df["runs_off_bat"] + df["extras"]

# 4. Ball number and overs
df["ball_number"] = df.groupby("match_id").cumcount() + 1
df["overs_completed"] = df["ball_number"] // 6 + (df["ball_number"] % 6) / 10

# 5. Current score
df["current_score"] = df.groupby("match_id")["runs"].cumsum()

# 6. Wickets lost
df["is_wicket"] = df["player_dismissed"].notna().astype(int)
df["wickets_lost"] = df.groupby("match_id")["is_wicket"].cumsum()

# 7. Target score
df1 = pd.read_csv("dataset.csv")
df1 = df1[df1["innings"] == 1].copy()
df1["runs"] = df1["runs_off_bat"] + df1["extras"]

targets = df1.groupby("match_id")["runs"].sum().reset_index()
targets["target_score"] = targets["runs"] + 1
targets = targets[["match_id", "target_score"]]

df = df.merge(targets, on="match_id", how="inner")

# 8. Final result label
final_score = df.groupby("match_id")["current_score"].max().reset_index(name="final_score")
df = df.merge(final_score, on="match_id")
df["win"] = (df["final_score"] >= df["target_score"]).astype(int)

# 9. Required run rate
TOTAL_OVERS = 20
df["remaining_overs"] = TOTAL_OVERS - df["overs_completed"]
df["remaining_runs"] = df["target_score"] - df["current_score"]
df["required_run_rate"] = (df["remaining_runs"] / df["remaining_overs"]).replace([np.inf, -np.inf], 0)

# 🔥 10. KEEP ONLY MEANINGFUL OVERS
df = df[df["overs_completed"] >= 6]

# 11. Final dataset
final_df = df[
    [
        "current_score",
        "overs_completed",
        "wickets_lost",
        "target_score",
        "required_run_rate",
        "win"
    ]
].dropna()

# 12. Train model
X = final_df.drop("win", axis=1)
y = final_df["win"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model Accuracy:", accuracy)

# 13. Save model
with open("win_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("win_model.pkl created successfully")
