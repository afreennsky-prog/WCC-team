# CareQueue — Smart Healthcare Queue Management

CareQueue is a Flask demo application for clinic appointment booking, queue visibility, specialty-based doctor discovery, and hospital operations. It includes a scikit-learn model for estimating patient waiting time and a rule-based patient Care Guide.

> **Demo software only.** This project is not a clinical system and is not ready for production or for storing real patient data. Its sample records and operational actions are illustrative.

## Contents

- [Features](#features)
- [Technology](#technology)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [Specialties](#specialties)
- [HTTP routes and API](#http-routes-and-api)
- [Data and persistence](#data-and-persistence)
- [Wait-time model](#wait-time-model)
- [Safety and limitations](#safety-and-limitations)
- [Troubleshooting](#troubleshooting)

## Features

- Patient registration and sign-in with hashed passwords.
- Doctor registration and sign-in, with a selected specialty.
- Appointment booking that checks doctor specialty and time-slot availability.
- Queue estimates and operational dashboard views.
- Doctor-to-patient assignment and patient lookup demo flows.
- Sample SMS/email reminder simulation.
- Hospital report page and CSV export.
- A floating Care Guide for clinic navigation and broad symptom-to-department routing.
- Emergency phrase detection that directs users to local emergency services.
- A bundled Random Forest model for estimated waiting times.

Some screens also display sample operational data. Reminder sending, queue escalation, and clinical operations are not connected to external hospital systems or real SMS/email providers.

## Technology

- Python and Flask
- HTML, CSS, and browser JavaScript
- JSON files for demo records
- pandas, NumPy, scikit-learn, and joblib for the wait-time model

## Project layout

```text
.
├── app.py                      # Flask app, routes, demo data and chat logic
├── requirements.txt            # Web app and model dependencies
├── bookings.json               # Persisted appointment bookings
├── users.json                  # Persisted patient accounts
├── doctors.json                # Persisted doctor accounts
├── index.html                  # Static frontend asset
├── script.js                   # Static frontend JavaScript
├── styles.css                  # Static frontend styles
├── templates/                  # Flask-rendered pages
│   ├── all_in_one.html
│   ├── booking.html
│   ├── doctor_login.html
│   ├── doctor_panel.html
│   ├── doctor_register.html
│   ├── landing.html
│   └── ...
└── ml/
    ├── dataset.csv             # Example training data
    ├── train_model.py          # Model training and evaluation
    ├── test_model.py           # Example model prediction
    └── waiting_time_model.pkl  # Bundled trained model
```

## Requirements

- Python 3.10 or later recommended.
- pip.
- The model artifact at `ml/waiting_time_model.pkl` must be present to start the app.

Install the dependencies from the project root:

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

```bat
.venv\Scripts\activate.bat
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Then install packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start

From the project root, run Flask bound to the local machine:

```bash
python -m flask --app app run --host 127.0.0.1 --port 5000
```

Open the landing page at:

<http://127.0.0.1:5000/>

The combined operations dashboard and Care Guide are at:

<http://127.0.0.1:5000/all-in-one>

Stop the development server with **Ctrl+C** in the terminal.

> Do not expose Flask's development server to the public internet. See [Safety and limitations](#safety-and-limitations) before using this project beyond a local demo.

## Using the app

1. Open `/` and choose a patient, doctor, booking, or all-in-one demo link.
2. Create a patient account from `/register`, then sign in from `/login`.
3. Book through `/booking`: choose a department, matching doctor, date, and available time.
4. Use `/doctor-register` to create a doctor login, or `/doctor-login` to sign in.
5. Visit `/doctor-panel` for the doctor-facing demo panel.
6. Open `/all-in-one` for the combined operations screen and floating Care Guide.
7. Use `/report` for the printable report or `/export/report.csv` to download the assignment CSV.

Patient and doctor passwords must be at least eight characters. Doctor registration requires a listed specialty. Accounts are saved locally in JSON files; this demo does not verify the identity or credentials of newly registered users.

## Specialties

Doctors can register under one of these departments:

| Department | General routing area |

| General Medicine | Common illnesses and symptoms |
| Cardiology | Heart and blood pressure |
| Neurology | Brain, nerves, and headaches |
| Pulmonology | Lungs, cough, and breathing |
| Endocrinology | Diabetes, thyroid, and hormones |
| Dermatology | Skin, hair, and nails |
| Pediatrics | Children's health |
| Gynecology | Women's health |
| Orthopedics | Bones, joints, and muscles |
| Ophthalmology | Eyes and vision |
| Psychiatry | Mental health |
| Dentistry | Teeth and oral health |

The Care Guide only lists doctors whose registered specialty matches the routed department. If none are registered for that specialty, it tells the user to contact the clinic rather than listing an unrelated doctor.

## HTTP routes and API

### Web pages

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Landing page |
| `GET` | `/login` | Patient sign-in page |
| `GET` | `/register` | Patient registration page |
| `GET` | `/booking` | Appointment booking page |
| `GET` | `/dashboard` | Queue and operational dashboard |
| `GET` | `/patient-portal` | Patient portal |
| `GET` | `/doctor` | Doctor view |
| `GET` | `/doctor-login` | Doctor sign-in page |
| `GET` | `/doctor-register` | Doctor account creation page |
| `GET` | `/doctor-panel` | Doctor panel |
| `GET` | `/all-in-one` | Combined operations dashboard with Care Guide |
| `GET` | `/admin` | Demo admin page |
| `GET` | `/report` | Printable report page |

### JSON and export endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/register` | Create a patient account |
| `POST` | `/api/login` | Sign in a patient |
| `POST` | `/api/logout` | Clear the current session |
| `POST` | `/api/doctor-register` | Create a doctor account |
| `POST` | `/api/doctor-login` | Sign in a doctor |
| `POST` | `/api/doctor-logout` | Clear the current session |
| `POST` | `/api/doctor-specialty` | Update the signed-in doctor's specialty |
| `GET` | `/api/doctors` | List doctors; optional `?department=Cardiology` filter |
| `GET` | `/api/queue` | Queue data and summary |
| `GET` | `/api/notifications` | Demo notifications |
| `GET` | `/api/clinical-overview` | Dashboard overview data |
| `GET` | `/api/patient-convenience` | Patient convenience features |
| `GET` | `/api/booking-insights` | Booking insights |
| `POST` | `/api/chat` | Get a Care Guide response |
| `GET` | `/api/bookings` | List bookings |
| `POST` | `/api/bookings` | Validate and create a booking |
| `POST` | `/api/triage` | Return a simple demo urgency classification |
| `GET` / `POST` | `/api/patient-search` | Search for a patient record |
| `GET` | `/api/patient-assignments` | List demo patient assignments |
| `POST` | `/api/patient-assignment` | Create or update an assignment |
| `GET` | `/api/reminders` | List demo reminders |
| `POST` | `/api/reminders` | Add a simulated reminder |
| `GET` | `/export/report.csv` | Download assignments as CSV |

The chat endpoint accepts JSON such as:

```json
{
  "message": "I have a skin concern. Which department should I contact?"
}
```

Chat questions must be between 1 and 500 characters. The endpoint returns a JSON object containing `success` and `answer` on success.

## Data and persistence

- `bookings.json` stores bookings. A sample booking is created if the file is empty.
- `users.json` stores patient account details and password hashes.
- `doctors.json` stores doctor account details and password hashes.
- Assignments, reminders created during use, and several dashboard lists are held in application memory and reset when the server restarts.
- The repository includes sample names and appointment data; treat them as fictional demo records.
- Back up these JSON files before resetting or replacing data. Do not commit real patient information or credentials.

## Wait-time model

The bundled model estimates wait time using:

- `patients_ahead`
- `avg_consultation_time`
- `urgent_patients`
- `day`
- `hour`

The example dataset is small and intended for demonstration, not medical or operational decision-making. To retrain the model from the `ml/` folder:

```bash
python train_model.py
python test_model.py
```

`train_model.py` evaluates a Random Forest regressor and writes `waiting_time_model.pkl` in the same folder. The app loads this artifact at startup. The artifact is a Python pickle; only load a model file from a trusted source. scikit-learn pickle files can also be incompatible across library versions. If loading reports a version warning or error, retrain the model in the same environment used to run the app.

## Safety and limitations

- The Care Guide is an offline keyword/rule-based navigation helper, not an AI medical consultant. It cannot diagnose, recommend treatment or medication, interpret test results, or judge the severity of a condition.
- Symptom routing is approximate and should not be used to make clinical decisions.
- The urgency endpoint is a simple keyword demonstration, not a validated triage system.
- If someone may be in immediate danger, contact local emergency services or go to an emergency department; do not wait for a chatbot response.
- Reminder delivery is simulated. There is no SMS, email, WhatsApp, payment, lab, EHR, pharmacy, or hospital-system integration.
- Demo user data is stored in local JSON files. This is not an appropriate storage design for real health information.
- This project is not production-hardened. Before any deployment, implement authorization for every protected route/API, CSRF protection, secure session-cookie settings, secrets management, input and output protections, audit logging, rate limiting, persistent database storage, backups, monitoring, and applicable healthcare privacy/security compliance.
- Set a strong `CAREQUEUE_SECRET_KEY` environment variable for Flask sessions. The application contains a development fallback secret and must not rely on it in any shared or deployed environment.

## Troubleshooting

### `ModuleNotFoundError` on startup

Activate the project virtual environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### The wait-time model file cannot be found

Confirm that `ml/waiting_time_model.pkl` exists. If it is missing, install the project dependencies and retrain from the model directory:

```bash
cd ml
python train_model.py
```

### Port 5000 is already in use

Stop the other local development server, or run on a different port:

```bash
python -m flask --app app run --host 127.0.0.1 --port 5001
```

### Model version warning

The installed scikit-learn version may differ from the version used to create the bundled pickle. Retrain `ml/waiting_time_model.pkl` using the active environment, or install the compatible scikit-learn version used to create the artifact.
