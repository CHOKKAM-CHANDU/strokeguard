from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load saved transformers and model
ct = joblib.load('column_transformer.pkl')
sc = joblib.load('standard_scaler.pkl')
model = joblib.load('xgboost_stroke_model.pkl')

# Label encoders (created ONCE)
le_gender = LabelEncoder()
le_gender.classes_ = np.array(['Female', 'Male'])

le_married = LabelEncoder()
le_married.classes_ = np.array(['No', 'Yes'])

le_residence = LabelEncoder()
le_residence.classes_ = np.array(['Rural', 'Urban'])

# FastAPI app
app = FastAPI()

# Allow CORS so frontend can call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow all frontends
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class StrokeData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

@app.post("/predict")
def predict_stroke(data: StrokeData):

    # Convert to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Encode categorical fields (Label Encoding)
    input_df['gender'] = le_gender.transform(input_df['gender'])
    input_df['ever_married'] = le_married.transform(input_df['ever_married'])
    input_df['Residence_type'] = le_residence.transform(input_df['Residence_type'])

    # Ensure correct column order
    expected_cols = [
        'gender', 'age', 'hypertension', 'heart_disease',
        'ever_married', 'work_type', 'Residence_type',
        'avg_glucose_level', 'bmi', 'smoking_status'
    ]
    input_df = input_df[expected_cols]

    # Apply ColumnTransformer
    transformed = ct.transform(input_df)

    # Scale numeric features
    scaled = sc.transform(transformed)

    # Predict
    prediction = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0][1]

    return {
        "stroke_prediction": int(prediction),
        "risk_probability": float(proba)
    }
