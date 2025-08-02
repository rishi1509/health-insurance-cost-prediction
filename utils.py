import pandas as pd
import joblib
import numpy as np

# Load once globally
scaler = joblib.load("models/scaler.pkl")
model_features = joblib.load("models/model_features.pkl")

# Load saved label encoders
le_sex = joblib.load("models/le_sex.pkl")
le_smoker = joblib.load("models/le_smoker.pkl")
le_region = joblib.load("models/le_region.pkl")

def preprocess_input(input_data: dict):
    # Convert dict to DataFrame
    df = pd.DataFrame([input_data])
    
    # Apply label encoding using saved encoders
    df['sex'] = le_sex.transform(df['sex'])
    df['smoker'] = le_smoker.transform(df['smoker'])
    df['region'] = le_region.transform(df['region'])
    
    # Ensure correct column order
    df = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    
    # Scale features
    scaled = scaler.transform(df)
    
    return scaled
