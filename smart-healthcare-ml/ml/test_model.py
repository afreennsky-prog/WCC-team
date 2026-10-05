from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "waiting_time_model.pkl"
model = joblib.load(MODEL_PATH)

patient = pd.DataFrame([{
    "patients_ahead": 5,
    "avg_consultation_time": 12,
    "urgent_patients": 1,
    "day": 2,
    "hour": 10
}])

prediction = model.predict(patient)[0]

print("Predicted waiting time:", round(prediction, 2), "minutes")
