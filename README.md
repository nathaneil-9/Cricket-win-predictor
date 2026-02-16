# 🏏 Cricket Match Win Probability Prediction Using Machine Learning

## 📌 Project Overview
This project predicts the **win probability of a cricket team during a live match** based on the current match situation.  
It uses **machine learning** on historical ball-by-ball cricket data to estimate the chances of winning in real time.

The system is designed for **T20 cricket matches** and also displays useful match statistics such as:
- Runs needed
- Balls remaining
- Current Run Rate (CRR)
- Required Run Rate (RRR)

---

## 🎯 Objective
The objective of this project is to:
- Predict whether the batting team will **win or lose**
- Display the **winning probability percentage**
- Provide match context using run rates and remaining resources

---

## 📂 Dataset
- **Type:** Ball-by-ball cricket dataset  
- **Source:** Public cricket datasets (Cricsheet / Kaggle)  
- **Description:**  
  The dataset contains detailed ball-level information such as runs scored, extras, wickets, and innings details.  
  From this raw data, match-level features were derived.

---

## 🧠 Features Used for Prediction
The following features are used by the machine learning model:

1. Current Score  
2. Overs Completed  
3. Wickets Lost  
4. Target Score  
5. Required Run Rate  

To improve accuracy and reduce randomness, predictions are made **after 5 overs**.

---

## ⚙️ Machine Learning Model
- **Algorithm:** Random Forest Classifier  
- **Number of Trees:** 300–400  
- **Reason for selection:**
  - Captures non-linear match patterns
  - Handles noisy sports data effectively
  - Provides higher accuracy than linear models

---

## 📈 Model Performance
- **Accuracy:** ~85%  
- Accuracy was improved by:
  - Removing early-over match noise
  - Adding Required Run Rate as a feature
  - Training on meaningful match situations only

---

## 🖥️ Application Interface
The project includes an **interactive Streamlit application** where users can:
- Enter live match details
- View runs needed and balls remaining
- See Current Run Rate (CRR) and Required Run Rate (RRR)
- Get real-time win probability prediction

---

## ▶️ How to Run the Project

### 1️⃣ Train the Model
```bash
python model_train.py
