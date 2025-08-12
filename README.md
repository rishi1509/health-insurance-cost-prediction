
# 💰 Health Insurance Cost Prediction using XGBoost

## 📌 Overview
This project predicts **medical insurance charges** based on personal and lifestyle factors such as age, BMI, smoking habits, and region.  
It uses **XGBoost Regressor** with a well-structured **machine learning pipeline** to achieve high prediction accuracy.



## 📊 Dataset
The dataset contains information about insurance holders:
- **age** — Age of the primary beneficiary
- **sex** — Gender (male/female)
- **bmi** — Body Mass Index
- **children** — Number of dependents
- **smoker** — Smoking status (yes/no)
- **region** — Residential region in the U.S.
- **charges** — Individual medical costs billed by health insurance (Target variable)



 🛠 Features
 Robust Feature Engineering
  - One-hot encoding for categorical features: `sex`, `smoker`, `region`
  - Scaling of numerical features: `age`, `bmi`, `children`
 Model Training
  - Compared multiple models: **Random Forest Regressor** & **XGBoost Regressor**
  - Final model: **XGBoost** with optimized hyperparameters
  Hyperparameter Tuning
  - Used `GridSearchCV` to optimize `learning_rate`, `max_depth`, `n_estimators`
  Performance
  - Achieved **R² Score > 0.88** on the test set



## 📦 Project Structure
health_insurance_cost_prediction/
│
├── models/
│ ├── best_model_xgboost.pkl # Trained XGBoost model
│ ├── model_features.pkl # Feature names for preprocessing
│ └── scaler.pkl # StandardScaler object
│
├── notebooks/
│ └── insurance_model_training.ipynb # Jupyter notebook for training
│
├── utils.py # Preprocessing and helper functions
├── dashboard.py # Streamlit app for predictions
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## 🚀 How to Run

### 1️⃣ Clone the repository

git clone https://github.com/yourusername/health-insurance-cost-prediction.git
cd health-insurance-cost-prediction
2️⃣ Install dependencies

pip install -r requirements.txt
3️⃣ Run the Streamlit app


streamlit run dashboard.py
🖥 Streamlit Web App
The app provides an interactive form to input:

Age

Gender

BMI

Number of children

Smoking status

Region

It then outputs predicted medical insurance charges.

📈 Example Prediction
Input:

Age: 35
Sex: male
BMI: 28.5
Children: 2
Smoker: no
Region: northwest
Output:

Predicted Insurance Charges: $8,745.32
🧠 Technologies Used
Python (pandas, numpy, scikit-learn, joblib)

XGBoost

Streamlit for interactive UI

Jupyter Notebook for training

Matplotlib / Seaborn for data visualization


