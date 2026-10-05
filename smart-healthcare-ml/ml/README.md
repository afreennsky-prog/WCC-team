# Smart Healthcare Queue - ML Model

This project trains a Random Forest Regression model to predict patient waiting time.

## Features
- patients_ahead
- avg_consultation_time
- urgent_patients
- day
- hour

## Target
- waiting_time (minutes)

## Run in VS Code

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
python train_model.py
python test_model.py
```

`train_model.py` creates `waiting_time_model.pkl`.

## Important
The included dataset is a small demonstration dataset for testing the code. For a real hackathon model, replace it with a larger, realistic historical dataset and validate the model carefully.
