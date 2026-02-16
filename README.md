# 🏏 Cricket Match Win Probability Prediction Using Machine Learning

## 📌 Project Overview
This project predicts the **win probability of a cricket team during a live match** based on the current match situation.  
It uses **machine learning** to analyze historical ball-by-ball cricket data and estimates the chances of winning in real time.

The system is designed mainly for **T20 matches** and provides additional match insights such as:
- Runs needed
- Balls remaining
- Current Run Rate (CRR)
- Required Run Rate (RRR)

---

## 🎯 Objective
To build a machine learning model that can:
- Predict whether the batting team will win or lose
- Display the **probability of winning** based on match conditions
- Help users understand match pressure and momentum

---

## 📂 Dataset
- **Type:** Ball-by-ball cricket dataset
- **Source:** Public cricket datasets (e.g. Cricsheet / Kaggle)
- **Key columns used:**
  - runs_off_bat
  - extras
  - innings
  - ball
  - player_dismissed

From these, match situation features were derived.

---

## 🧠 Features Used for Prediction
The final model uses the following features:

1. Current Score  
2. Overs Completed  
3. Wickets Lost  
4. Target Score  
5. Required Run Rate  

To reduce noise and improve accuracy, predictions are made **after 5 overs**.

---

## ⚙️ Machine Learning Model
- **Algorithm:** Random Forest Classifier  
- **Number of Trees:** 300–400  
- **Reason for selection:**
  - Handles non-linear patterns
  - Performs well on sports data
  - Improves accuracy compared to linear models

---

## 📈 Model Performance
- **Accuracy:** ~85%  
- Accuracy was improved by:
  - Removing early-over noise
  - Using match pressure features like Required Run Rate
  - Sampling meaningful match situations

---

## 🖥️ Application Interface
The project includes a **Streamlit-based interactive application** where users can:
- Enter live match details
- View match statistics (CRR, RRR, balls remaining)
- Get real-time win probability prediction

---

## ▶️ How to Run the Project

### 1️⃣ Train the Model
```bash
python
