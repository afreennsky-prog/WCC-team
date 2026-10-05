from __future__ import annotations

import csv
import json
import os
import random
import re
from datetime import datetime
from io import StringIO
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, session
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ml" / "waiting_time_model.pkl"
if os.environ.get("VERCEL"):
    os.environ.setdefault("CAREQUEUE_DATA_DIR", "/tmp/carequeue-data")
    if not os.environ.get("CAREQUEUE_SECRET_KEY"):
        raise RuntimeError("Set CAREQUEUE_SECRET_KEY in the Vercel project environment.")
DATA_DIR = Path(os.environ.get("CAREQUEUE_DATA_DIR", BASE_DIR)).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = DATA_DIR / "bookings.json"
USER_FILE = DATA_DIR / "users.json"
DOCTOR_FILE = DATA_DIR / "doctors.json"

DOCTOR_SPECIALTIES = (
    "General Medicine",
    "Cardiology",
    "Neurology",
    "Pulmonology",
    "Endocrinology",
    "Dermatology",
    "Pediatrics",
    "Gynecology",
    "Orthopedics",
    "Ophthalmology",
    "Psychiatry",
    "Dentistry",
)

SPECIALTY_DESCRIPTIONS = {
    "Cardiology": "heart and blood pressure",
    "Neurology": "brain, nerves, and headaches",
    "General Medicine": "common illnesses and symptoms",
    "Pulmonology": "lungs, cough, and breathing",
    "Endocrinology": "diabetes, thyroid, and hormones",
    "Dermatology": "skin, hair, and nails",
    "Pediatrics": "children's health",
    "Gynecology": "women's health",
    "Orthopedics": "bones, joints, and muscles",
    "Ophthalmology": "eyes and vision",
    "Psychiatry": "mental health",
    "Dentistry": "teeth and oral health",
}

DEFAULT_DOCTOR_SPECIALTIES = {
    "Dr. Aisha Patel": "General Medicine",
    "Dr. Rajesh Kumar": "Cardiology",
    "Dr. Anika Shah": "Dermatology",
    "Dr. Neha Gupta": "Orthopedics",
}

SPECIALTY_KEYWORDS = {
    "Cardiology": {"cardiology", "cardiologist", "cardio", "heart", "cardiac", "blood pressure", "hypertension", "palpitation", "palpitations", "chest pressure"},
    "Neurology": {"neurology", "neurologist", "brain", "nerve", "nerves", "headache", "headaches", "migraine", "migraines", "numbness", "tingling", "tremor", "tremors", "memory problem", "dizziness"},
    "General Medicine": {"general medicine", "general physician", "general practitioner", "primary care", "general doctor", "common illness", "stomach", "stomach ache", "stomachache", "nausea", "vomiting", "diarrhea", "diarrhoea", "constipation", "indigestion", "earache", "ear pain", "sore throat", "fatigue", "allergy", "allergies", "allergic", "infection", "urinary", "urine", "burning urination", "swelling"},
    "Pulmonology": {"pulmonology", "pulmonologist", "lung", "lungs", "cough", "coughing", "breathing", "respiratory", "asthma", "wheezing", "phlegm"},
    "Endocrinology": {"endocrinology", "endocrinologist", "diabetes", "diabetic", "thyroid", "hormone", "hormones", "blood sugar", "insulin"},
    "Pediatrics": {"pediatrics", "pediatrician", "paediatrician", "child", "children", "infant", "baby", "kid", "kids"},
    "Dermatology": {"dermatology", "dermatologist", "skin", "rash", "acne", "hair", "hair loss", "nails", "eczema", "itch", "itchy", "itching", "hives"},
    "Gynecology": {
        "gynecology", "gynecologist", "gynecological", "gynecologic",
        "gynaecology", "gynaecologist", "gynaecological", "gynaecologic",
        "women's health", "womens health", "women s health", "women health", "female health",
        "menstrual", "menstruation", "menstrual cramps", "fertility",
    },
    "Orthopedics": {"orthopedics", "orthopedist", "orthopedic", "orthopaedic", "bone", "bones", "joint", "joints", "muscle", "muscles", "knee", "fracture", "back pain", "backache", "ankle", "shoulder", "wrist", "sprain"},
    "Ophthalmology": {"ophthalmology", "ophthalmologist", "eye", "eyes", "vision", "sight", "blurry vision", "blurred vision", "eyelid"},
    "Psychiatry": {"psychiatry", "psychiatrist", "mental health", "anxiety", "anxious", "depression", "depressed", "sad", "hopeless", "panic attack", "panic attacks", "mood", "insomnia", "sleep problem"},
    "Dentistry": {"dentistry", "dentist", "dental", "tooth", "toothache", "teeth", "gum", "gums", "oral", "jaw pain", "cavity", "cavities"},
}

DOCTOR_SCHEDULE = {
    "Dr. Aisha Patel": ["09:00", "09:30", "10:00", "10:30", "11:00", "12:00", "14:00", "15:00", "16:00"],
    "Dr. Rajesh Kumar": ["09:15", "09:45", "10:15", "10:45", "11:15", "13:00", "14:30", "15:30"],
    "Dr. Anika Shah": ["08:30", "09:00", "09:30", "10:00", "11:30", "12:30", "14:00", "15:00"],
    "Dr. Neha Gupta": ["09:30", "10:00", "10:30", "11:00", "12:00", "14:00", "15:30", "16:00"],
}

app = Flask(__name__, static_folder="public", static_url_path="")
app.secret_key = os.environ.get("CAREQUEUE_SECRET_KEY", "carequeue-development-only-change-before-deploy")
model = joblib.load(MODEL_PATH)


def load_users():
    if not USER_FILE.exists():
        return []
    with USER_FILE.open("r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            return []
    return data if isinstance(data, list) else []


def save_users(users):
    with USER_FILE.open("w", encoding="utf-8") as file:
        json.dump(users, file, indent=2)


def load_doctors():
    if not DOCTOR_FILE.exists():
        return []
    with DOCTOR_FILE.open("r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            return []
    return data if isinstance(data, list) else []


def save_doctors(doctors):
    with DOCTOR_FILE.open("w", encoding="utf-8") as file:
        json.dump(doctors, file, indent=2)


users = load_users()


def load_bookings():
    if not DATA_FILE.exists():
        DATA_FILE.write_text("[]", encoding="utf-8")
    with DATA_FILE.open("r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            data = []
    return data if isinstance(data, list) else []


def save_bookings(bookings):
    with DATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(bookings, file, indent=2)


def normalize_priority(priority):
    value = (priority or "routine").lower().strip()
    if value in {"emergency", "critical"}:
        return "emergency"
    if value in {"urgent", "high"}:
        return "urgent"
    return "routine"


def build_queue_summary(queue):
    if not queue:
        return {"appointments_managed": 0, "efficiency": 0, "urgent_cases": 0, "average_wait": 0, "overbook_risk": 0}

    avg_wait = round(sum(item["estimated_wait"] for item in queue) / len(queue), 1)
    urgent_cases = sum(1 for item in queue if item.get("priority") in {"urgent", "emergency"})
    overbook_risk = min(100, round((len(queue) / 6) * 100))
    return {
        "appointments_managed": 12000 + len(queue) * 42,
        "efficiency": max(70, 92 - len(queue)),
        "urgent_cases": urgent_cases,
        "average_wait": avg_wait,
        "overbook_risk": overbook_risk,
    }


def get_next_available_slot(doctor_name, date_value):
    date_value = date_value or datetime.today().strftime("%Y-%m-%d")
    existing = {entry.get("time") for entry in bookings if entry.get("doctor") == doctor_name and entry.get("date") == date_value}
    for slot in DOCTOR_SCHEDULE.get(doctor_name, ["09:00", "10:00", "11:00", "14:00", "15:00"]):
        if slot not in existing:
            return slot
    return "09:00"


def get_smart_slot_recommendations(specialty=None, date_value=None):
    recommendation_date = date_value or datetime.today().strftime("%Y-%m-%d")
    try:
        parsed_date = datetime.strptime(recommendation_date, "%Y-%m-%d").date()
    except ValueError:
        return []

    recommendations = []
    eligible_doctors = [
        doctor for doctor in doctor_accounts
        if doctor.get("specialty") in DOCTOR_SPECIALTIES
        and (not specialty or doctor.get("specialty") == specialty)
    ]
    for doctor in eligible_doctors:
        doctor_name = doctor["name"]
        slots = DOCTOR_SCHEDULE.get(doctor_name, ["09:00", "10:00", "11:00", "14:00", "15:00"])
        occupied = {entry.get("time") for entry in bookings if entry.get("doctor") == doctor_name and entry.get("date") == recommendation_date}
        available_slots = [slot for slot in slots if slot not in occupied]
        if parsed_date == datetime.today().date():
            current_time = datetime.now().strftime("%H:%M")
            available_slots = [slot for slot in available_slots if slot >= current_time]
        if available_slots:
            recommendations.append({
                "doctor": doctor_name,
                "date": recommendation_date,
                "time": available_slots[0],
                "available_slots": available_slots,
                "label": "Best slot",
            })
    return recommendations


def validate_booking(payload):
    patient_name = (payload.get("patient_name") or payload.get("name") or "").strip()
    doctor = (payload.get("doctor") or "").strip()
    date_value = (payload.get("date") or "").strip()
    time_value = (payload.get("time") or "").strip()

    if not patient_name or not doctor or not date_value:
        return False, "Patient name, doctor, and date are required."
    doctor_record = get_doctor_by_name(doctor)
    if not doctor_record:
        return False, "Please choose a doctor from the directory."
    if doctor_record.get("specialty") != payload.get("department"):
        return False, f"{doctor} is listed under {doctor_record.get('specialty', 'an unspecified specialty')}. Choose a doctor who matches the selected department."
    if time_value not in DOCTOR_SCHEDULE.get(doctor, ["09:00", "10:00", "11:00", "14:00", "15:00"]):
        return False, "Please select an appointment time from the doctor's available slots."

    for entry in bookings:
        same_patient = (entry.get("name") or "").strip().lower() == patient_name.lower()
        same_doctor = (entry.get("doctor") or "").strip() == doctor
        same_date = (entry.get("date") or "") == date_value
        if same_patient and same_doctor and same_date:
            return False, "This patient already has an appointment with the same doctor on that date."
        if same_doctor and same_date and (entry.get("time") or "") == time_value:
            return False, "This doctor is already booked during the selected time slot."

    return True, "Booked successfully."


queue_data = [
    {
        "name": "Riya Shah",
        "doctor": "Dr. Aisha Patel",
        "queue_id": "Q-2418",
        "status": "In queue",
        "patients_ahead": 7,
        "avg_consultation_time": 12,
        "urgent_patients": 1,
        "day": 2,
        "hour": 10,
        "service": "General Medicine",
        "priority": "urgent",
    },
    {
        "name": "Arjun Mehta",
        "doctor": "Dr. Rajesh Kumar",
        "queue_id": "Q-2419",
        "status": "Lab review",
        "patients_ahead": 3,
        "avg_consultation_time": 15,
        "urgent_patients": 0,
        "day": 2,
        "hour": 10,
        "service": "Cardiology",
        "priority": "routine",
    },
    {
        "name": "Sneha Verma",
        "doctor": "Dr. Anika Shah",
        "queue_id": "Q-2420",
        "status": "Ready",
        "patients_ahead": 1,
        "avg_consultation_time": 9,
        "urgent_patients": 0,
        "day": 2,
        "hour": 11,
        "service": "Dermatology",
        "priority": "routine",
    },
    {
        "name": "Imran Khan",
        "doctor": "Dr. Neha Gupta",
        "queue_id": "Q-2421",
        "status": "Follow-up",
        "patients_ahead": 4,
        "avg_consultation_time": 14,
        "urgent_patients": 1,
        "day": 2,
        "hour": 12,
        "service": "Orthopedics",
        "priority": "urgent",
    },
]

bookings = load_bookings()
if not bookings:
    bookings = [
        {
            "name": "Riya Shah",
            "department": "General Medicine",
            "doctor": "Dr. Aisha Patel",
            "date": datetime.today().strftime("%Y-%m-%d"),
            "time": "10:30",
            "priority": "urgent",
            "email": "riya@example.com",
            "phone": "9876543210",
        }
    ]
    save_bookings(bookings)

notifications = [
    {"title": "Emergency triage alert", "message": "Two urgent patients require immediate assessment.", "time": "2 min ago", "severity": "high"},
    {"title": "Lab results ready", "message": "Blood work for Arjun Mehta is ready for review.", "time": "8 min ago", "severity": "info"},
    {"title": "Pharmacy queue update", "message": "Prescription refill queue is increasing by 18%.", "time": "15 min ago", "severity": "medium"},
]

care_reminders = [
    {"patient": "Riya Shah", "type": "Medication", "message": "Take blood pressure medicine after breakfast and confirm refill request.", "due": "Morning"},
    {"patient": "Sneha Verma", "type": "Follow-up", "message": "Review dermatologist follow-up note before the end of the day.", "due": "Today"},
    {"patient": "Imran Khan", "type": "Lab", "message": "MRI report is ready for discussion in the next consultation window.", "due": "Tomorrow"},
]

DOCTOR_DISPLAY_NAMES = {
    "aisha": "Dr. Aisha Patel",
    "rajesh": "Dr. Rajesh Kumar",
    "anika": "Dr. Anika Shah",
    "neha": "Dr. Neha Gupta",
}

doctor_accounts = load_doctors()
if not doctor_accounts:
    doctor_accounts = [
        {
            "username": username,
            "name": DOCTOR_DISPLAY_NAMES[username],
            "specialty": DEFAULT_DOCTOR_SPECIALTIES[DOCTOR_DISPLAY_NAMES[username]],
            "password_hash": generate_password_hash("care123"),
        }
        for username in DOCTOR_DISPLAY_NAMES
    ]
    save_doctors(doctor_accounts)
else:
    updated_doctor_accounts = False
    for doctor_account in doctor_accounts:
        if not doctor_account.get("specialty"):
            doctor_account["specialty"] = DEFAULT_DOCTOR_SPECIALTIES.get(
                doctor_account.get("name"), "Unspecified"
            )
            updated_doctor_accounts = True
    if updated_doctor_accounts:
        save_doctors(doctor_accounts)


def get_doctors_for_specialty(specialty):
    return [
        doctor for doctor in doctor_accounts
        if doctor.get("specialty", "Unspecified").casefold() == specialty.casefold()
    ]


def get_doctor_by_name(name):
    return next(
        (doctor for doctor in doctor_accounts if doctor.get("name", "").casefold() == name.casefold()),
        None,
    )

assigned_patients = [
    {"patient": "Riya Shah", "doctor": "Dr. Aisha Patel", "status": "In consultation", "room": "Room 2", "priority": "urgent"},
    {"patient": "Arjun Mehta", "doctor": "Dr. Rajesh Kumar", "status": "Awaiting lab", "room": "Room 5", "priority": "routine"},
    {"patient": "Sneha Verma", "doctor": "Dr. Anika Shah", "status": "Ready for review", "room": "Room 3", "priority": "routine"},
    {"patient": "Imran Khan", "doctor": "Dr. Neha Gupta", "status": "Escalated", "room": "Observation", "priority": "urgent"},
]

patient_convenience_features = [
    {
        "title": "Appointment reminders",
        "detail": "Get SMS and WhatsApp reminders 24 hours before your visit, plus check-in alerts.",
        "status": "Active",
    },
    {
        "title": "Digital prescriptions",
        "detail": "View your prescription, renewal timing, and medicine instructions in one place.",
        "status": "Ready",
    },
    {
        "title": "Virtual follow-ups",
        "detail": "Book quick follow-up consults without waiting in a long physical queue.",
        "status": "New",
    },
    {
        "title": "Lab report access",
        "detail": "Track results and share them with your doctor in a single click.",
        "status": "Updated",
    },
    {
        "title": "Billing support",
        "detail": "See treatment estimates, payment status, and invoice summary before checkout.",
        "status": "Live",
    },
    {
        "title": "Care guidance",
        "detail": "Get preventive health tips and post-visit recovery instructions from the clinic.",
        "status": "Helpful",
    },
]


def get_wait_prediction(patient):
    feature_row = pd.DataFrame([
        {
            "patients_ahead": patient["patients_ahead"],
            "avg_consultation_time": patient["avg_consultation_time"],
            "urgent_patients": patient["urgent_patients"],
            "day": patient["day"],
            "hour": patient["hour"],
        }
    ])
    prediction = model.predict(feature_row)[0]
    return max(float(prediction), 5.0)


def build_live_queue(queue):
    live_queue = []
    for patient in queue:
        wait = get_wait_prediction(patient)
        jitter = random.uniform(-3.0, 3.0)
        wait = round(max(5.0, wait + jitter), 1)
        live_queue.append(
            {
                "name": patient["name"],
                "doctor": patient["doctor"],
                "queue_id": patient["queue_id"],
                "status": patient["status"],
                "service": patient["service"],
                "patients_ahead": patient["patients_ahead"],
                "estimated_wait": wait,
                "priority": patient["priority"],
            }
        )
    return live_queue


def lookup_patient_record(name):
    normalized = (name or "").strip().lower()
    if not normalized:
        return None

    booking_match = next(
        (entry for entry in bookings if (entry.get("name") or "").strip().lower() == normalized),
        None,
    )
    if booking_match:
        return {
            "kind": "booking",
            "name": booking_match.get("name"),
            "doctor": booking_match.get("doctor"),
            "department": booking_match.get("department"),
            "date": booking_match.get("date"),
            "time": booking_match.get("time"),
            "priority": booking_match.get("priority", "routine"),
            "status": booking_match.get("booking_status", "Confirmed"),
        }

    queue_match = next(
        (entry for entry in queue_data if (entry.get("name") or "").strip().lower() == normalized),
        None,
    )
    if queue_match:
        return {
            "kind": "queue",
            "name": queue_match.get("name"),
            "doctor": queue_match.get("doctor"),
            "department": queue_match.get("service"),
            "status": queue_match.get("status"),
            "queue_id": queue_match.get("queue_id"),
            "priority": queue_match.get("priority", "routine"),
            "estimated_wait": round(max(5.0, get_wait_prediction(queue_match)), 1),
        }

    return None


def build_health_dashboard():
    live_queue = build_live_queue(queue_data)
    return {
        "summary": build_queue_summary(live_queue),
        "care_reminders": care_reminders,
        "doctor_capacity": [
            {"doctor": "Dr. Aisha Patel", "slots": 5, "utilization": 82},
            {"doctor": "Dr. Rajesh Kumar", "slots": 4, "utilization": 76},
            {"doctor": "Dr. Anika Shah", "slots": 6, "utilization": 68},
            {"doctor": "Dr. Neha Gupta", "slots": 5, "utilization": 88},
        ],
    }


@app.get("/")
def index():
    return render_template("landing.html")


@app.get("/login")
def login():
    return render_template("login.html")


@app.post("/api/register")
def register_api():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    phone = (payload.get("phone") or "").strip()

    if not name or not email or not password:
        return jsonify({"success": False, "message": "Name, email, and password are required."}), 400
    if len(password) < 8:
        return jsonify({"success": False, "message": "Use a password with at least 8 characters."}), 400
    if any(user.get("email") == email for user in users):
        return jsonify({"success": False, "message": "An account with this email already exists. Sign in instead."}), 409

    account = {
        "name": name,
        "email": email,
        "phone": phone,
        "age": (payload.get("age") or "").strip(),
        "gender": (payload.get("gender") or "").strip(),
        "department": (payload.get("department") or "General Medicine").strip(),
        "password_hash": generate_password_hash(password),
    }
    users.append(account)
    save_users(users)
    session.clear()
    session["user_email"] = email
    session["user_name"] = name
    session["user_role"] = "patient"
    return jsonify({"success": True, "message": "Account created. You are now signed in.", "redirect": "/patient-portal"}), 201


@app.post("/api/login")
def login_api():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    account = next((user for user in users if user.get("email") == email), None)

    if not account or not check_password_hash(account.get("password_hash", ""), password):
        return jsonify({"success": False, "message": "Email or password is incorrect. Register first if you do not have an account."}), 401

    session.clear()
    session["user_email"] = account["email"]
    session["user_name"] = account["name"]
    session["user_role"] = "patient"
    return jsonify({"success": True, "message": "Signed in successfully.", "redirect": "/patient-portal"})


@app.post("/api/logout")
def logout_api():
    session.clear()
    return jsonify({"success": True, "redirect": "/"})


@app.get("/register")
def register():
    return render_template(
        "register.html",
        specialties=DOCTOR_SPECIALTIES,
        specialty_descriptions=SPECIALTY_DESCRIPTIONS,
    )


@app.get("/booking")
def booking():
    return render_template(
        "booking.html",
        specialties=DOCTOR_SPECIALTIES,
        specialty_descriptions=SPECIALTY_DESCRIPTIONS,
    )


@app.get("/dashboard")
def dashboard():
    live_queue = build_live_queue(queue_data)
    overview = build_health_dashboard()
    return render_template(
        "dashboard.html",
        patients=live_queue,
        summary=overview["summary"],
        recommendations=get_smart_slot_recommendations(),
        reminders=overview["care_reminders"],
        capacity=overview["doctor_capacity"],
    )


@app.get("/patient-portal")
def patient_portal():
    return render_template("patient_portal.html", convenience_features=patient_convenience_features)


@app.get("/doctor")
def doctor():
    return render_template("doctor.html", patients=queue_data, assignments=assigned_patients)


@app.get("/doctor-login")
def doctor_login():
    return render_template("doctor_login.html")


@app.get("/doctor-register")
def doctor_register():
    return render_template(
        "doctor_register.html",
        specialties=DOCTOR_SPECIALTIES,
        specialty_descriptions=SPECIALTY_DESCRIPTIONS,
    )


@app.get("/all-in-one")
def all_in_one():
    return render_template(
        "all_in_one.html",
        doctor_name=session.get("doctor_name", ""),
    )


@app.get("/doctor-panel")
def doctor_panel():
    doctor_name = session.get("doctor_name") or "Dr. Aisha Patel"
    filtered = [item for item in assigned_patients if item["doctor"] == doctor_name] or assigned_patients
    doctor_names = [account["name"] for account in doctor_accounts]
    doctor_account = get_doctor_by_name(doctor_name)
    return render_template(
        "doctor_panel.html",
        assignments=filtered,
        doctor_name=doctor_name,
        doctor_names=doctor_names,
        doctor_specialty=doctor_account.get("specialty", "Unspecified") if doctor_account else "Unspecified",
        specialties=DOCTOR_SPECIALTIES,
        specialty_descriptions=SPECIALTY_DESCRIPTIONS,
    )


@app.get("/admin")
def admin():
    return render_template("admin.html", notifications=notifications)


@app.get("/api/queue")
def queue_api():
    live_queue = build_live_queue(queue_data)
    return jsonify(
        {
            "summary": build_queue_summary(live_queue),
            "patients": live_queue,
            "alerts": [
                {"text": "Two emergency cases awaiting review", "severity": "high"},
                {"text": "Clinic efficiency above target this morning", "severity": "low"},
            ],
        }
    )


@app.get("/api/notifications")
def notifications_api():
    return jsonify({"notifications": notifications})


@app.post("/api/chat")
def patient_chat_api():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()
    if not question:
        return jsonify({"success": False, "message": "Please enter a question."}), 400
    if len(question) > 500:
        return jsonify({"success": False, "message": "Please keep your question under 500 characters."}), 400

    answer = answer_patient_question(question)
    return jsonify({"success": True, "answer": answer})


def normalize_chat_text(value):
    return re.sub(r"[^a-z0-9\s]", " ", value.lower())


def answer_patient_question(question):
    text = normalize_chat_text(question)
    words = set(text.split())

    emergency_phrases = (
        "chest pain", "pain in my chest", "pain in chest", "chest pressure", "can't breathe", "cannot breathe",
        "difficulty breathing", "shortness of breath", "not breathing",
        "severe bleeding", "uncontrolled bleeding", "stroke symptoms",
        "face drooping", "sudden weakness", "sudden confusion", "unconscious", "passed out", "fainting", "overdose",
        "poisoning", "heart attack", "anaphylaxis", "severe allergic reaction",
        "sudden vision loss", "severe sudden headache", "suicide", "suicidal",
        "seizure", "self harm", "hurt myself", "harm myself", "kill myself",
        "want to die", "ending my life",
    )
    emergency_match = False
    for phrase in emergency_phrases:
        match = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text)
        if not match:
            continue
        preceding_text = text[max(0, match.start() - 32):match.start()]
        if re.search(r"\b(?:no|not|without|never|don't|doesn't|denies)\s+(?:\w+\s+){0,2}$", preceding_text):
            continue
        emergency_match = True
        break

    if emergency_match:
        return (
            "If someone may be in immediate danger, contact your local emergency number "
            "or go to the nearest emergency department now. Do not wait for a chatbot reply. "
            "This service cannot assess or diagnose emergencies."
        )

    requested_specialty = next(
        (
            specialty for specialty, keywords in SPECIALTY_KEYWORDS.items()
            if any(
                re.search(
                    rf"(?<!\w){re.escape(normalize_chat_text(keyword).strip())}(?!\w)",
                    text,
                )
                for keyword in keywords
            )
        ),
        None,
    )
    specialty_request_words = {
        "doctor", "doctors", "specialist", "specialists", "treatment",
        "treat", "treats", "treating", "handle", "handles", "see", "visit",
        "appointment", "book", "booking", "who", "which", "for", "need",
        "looking", "cardiologist", "dermatologist", "pediatrician",
        "paediatrician", "orthopedic", "orthopaedic", "neurologist",
        "pulmonologist", "endocrinologist", "gynecologist", "gynaecologist",
        "ophthalmologist", "psychiatrist", "dentist", "cardiology",
        "dermatology", "pediatrics", "paediatrics", "orthopedics",
        "orthopaedics", "ophthalmology", "psychiatry", "dentistry",
    }
    symptom_context_words = {
        "symptom", "symptoms", "problem", "problems", "issue", "issues",
        "concern", "concerns", "experiencing", "suffering", "have", "has",
        "pain", "headache", "headaches", "migraine", "migraines", "cough",
        "coughing", "rash", "acne", "itchy", "itching", "vision", "blurred",
        "diabetes", "diabetic", "thyroid", "breathing", "asthma", "joint",
        "joints", "bone", "bones", "muscle", "muscles", "tooth", "toothache",
        "teeth", "pregnant", "pregnancy", "period", "periods", "fever",
        "nausea", "vomiting", "cold", "flu", "stomach", "diarrhea",
        "diarrhoea", "eye", "eyes", "eyelid", "palpitation", "palpitations",
        "hurts", "hurt", "blood", "sugar", "high", "low", "skin", "hair",
        "nails", "anxiety", "depression", "mental", "oral", "constipation",
        "indigestion", "earache", "ear", "throat", "fatigue", "tired",
        "weakness", "weak", "numbness", "numb", "tingling", "tremor",
        "tremors", "memory", "wheezing", "phlegm", "sore", "sprain",
        "ankle", "shoulder", "wrist", "back", "jaw", "cavity", "cavities",
        "urine", "urinary", "urination", "burning", "itch", "swelling",
        "swollen", "vomit", "stomachache", "cramp", "cramps", "bleeding",
        "allergy", "allergies", "allergic", "infection", "menstrual", "menstruation",
        "fertility", "panic", "mood", "anxious", "depressed", "sad",
        "hopeless", "itchy", "itching", "dizzy",
        "dizziness", "eczema", "hives", "hair", "insomnia", "sleep",
    }
    if not requested_specialty and words & symptom_context_words:
        if words & {"child", "children", "infant", "baby", "kid", "kids"}:
            requested_specialty = "Pediatrics"
        elif words & {"pregnant", "pregnancy", "period", "periods"}:
            requested_specialty = "Gynecology"
        elif words & {"blood", "sugar", "diabetes", "diabetic", "thyroid", "hormone", "hormones"}:
            requested_specialty = "Endocrinology"
        elif words & {"anxiety", "depression", "depressed", "sad", "hopeless", "mental", "panic", "insomnia", "sleep"}:
            requested_specialty = "Psychiatry"
        elif words & {"eye", "eyes", "vision", "blurred", "eyelid"}:
            requested_specialty = "Ophthalmology"
        elif words & {"tooth", "toothache", "teeth", "gum", "gums", "oral", "jaw", "cavity", "cavities"}:
            requested_specialty = "Dentistry"
        elif words & {"heart", "cardiac", "palpitation", "palpitations", "hypertension"}:
            requested_specialty = "Cardiology"
        elif words & {"headache", "headaches", "migraine", "migraines", "numbness", "numb", "tingling", "tremor", "tremors", "memory"}:
            requested_specialty = "Neurology"
        elif words & {"lung", "lungs", "cough", "coughing", "breathing", "asthma", "wheezing", "phlegm"}:
            requested_specialty = "Pulmonology"
        elif words & {"back", "joint", "joints", "bone", "bones", "muscle", "muscles", "knee", "fracture", "ankle", "shoulder", "wrist", "sprain"}:
            requested_specialty = "Orthopedics"
        else:
            requested_specialty = "General Medicine"
    medical_intent = bool(words & specialty_request_words) or bool(words & symptom_context_words)
    if requested_specialty and medical_intent:
        matching_doctors = get_doctors_for_specialty(requested_specialty)
        if matching_doctors:
            doctor_names = ", ".join(doctor["name"] for doctor in matching_doctors)
            return (
                f"Based on the area you described, {requested_specialty} may be an appropriate department "
                f"to contact. Doctors listed: {doctor_names}. You can view appointment options on the "
                f"Booking page. This is general routing guidance only—not a diagnosis or treatment plan."
            )
        return (
            f"Based on the area you described, {requested_specialty} may be relevant, but no doctor in "
            "that specialty is currently listed here. Please contact the clinic to find an appropriate "
            "clinician. This is not a diagnosis; seek urgent care if symptoms are severe or worsening."
        )

    medical_terms = {
        "symptom", "symptoms", "diagnose", "diagnosis", "treatment", "medicine",
        "medication", "dose", "dosage", "prescription", "pain", "fever", "rash",
        "infection", "pregnant", "pregnancy", "allergy", "allergies", "allergic", "side",
        "effect", "blood", "pressure", "sick", "hurt", "wound", "antibiotic",
        "antibiotics", "headache", "headaches", "migraine", "migraines",
        "cough", "coughing", "palpitation", "palpitations", "sugar",
        "hurts", "toothache",
    }
    if words & medical_terms:
        return (
            "I can help with CareQueue and general clinic navigation, but I can't diagnose "
            "or recommend medicines, doses, or treatment. Please contact a licensed clinician "
            "for symptoms, medication questions, pregnancy, allergies, or test interpretation. "
            "For severe or rapidly worsening symptoms, contact emergency services."
        )

    if words & {"hello", "hi", "hey", "good"}:
        return (
            "Hello! I can help with booking, appointments, doctor listings, queue estimates, "
            "registration, and using the patient portal. What would you like help with?"
        )

    if words & {"insurance", "payment", "payments", "bill", "billing", "cost", "price", "fee", "fees"}:
        return (
            "Insurance, payment methods, and fees are clinic-specific and are not configured in this demo. "
            "Please confirm costs and coverage with reception or your insurer before your visit."
        )

    if words & {"human", "person", "staff", "receptionist", "reception"}:
        return (
            "This chatbot can't connect you to clinic staff. Please call or visit your clinic's reception "
            "using the contact details they provided. If it is an emergency, contact emergency services."
        )

    if words & {"prepare", "preparing", "bring", "arrival", "arrive", "early", "before"}:
        return (
            "Check your appointment date and time before travelling. Bring any items or records your clinic "
            "specifically requested, and contact reception if you need accessibility or arrival instructions. "
            "This demo doesn't have clinic-specific preparation rules."
        )

    if (
        words & {"doctor", "doctors", "physician", "physicians", "specialist", "specialists"}
        and not words & {"book", "booking", "schedule", "appointment", "appointments"}
    ):
        specialist_list = "; ".join(
            f"{specialty}: {', '.join(doctor['name'] for doctor in get_doctors_for_specialty(specialty))}"
            for specialty in DOCTOR_SPECIALTIES
            if get_doctors_for_specialty(specialty)
        )
        if words & {"available", "availability", "slot", "slots", "when", "time"}:
            suggestions = get_smart_slot_recommendations()
            if suggestions:
                listed_slots = "; ".join(
                    f"{item['doctor']} at {item['time']}" for item in suggestions[:4]
                )
                return (
                    f"Doctors with upcoming demo slots for {suggestions[0]['date']}: {listed_slots}. "
                    "Open Booking to choose a doctor and confirm the time."
                )
            return "No future demo slots are listed for today. Please check another date in Booking or contact the clinic."
        return (
            f"Doctors are listed by specialty: {specialist_list or 'No verified specialties are listed yet.'} "
            "Tell me the treatment area or department you need, and I can show matching doctors. "
            "This is directory guidance, not a diagnosis."
        )

    if words & {"book", "booking", "schedule", "appointment", "appointments", "slot", "slots", "available"}:
        if words & {"cancel", "cancelled", "cancellation", "reschedule", "change", "move"}:
            return (
                "This demo does not support cancelling or rescheduling yet. Please contact the "
                "clinic reception so staff can update your appointment safely."
            )
        if words & {"my", "mine", "check", "status", "when", "where"} and session.get("user_email"):
            patient_booking = next(
                (
                    booking_item for booking_item in bookings
                    if (booking_item.get("email") or "").strip().lower() == session["user_email"]
                ),
                None,
            )
            if patient_booking:
                return (
                    f"Your latest appointment on file is with {patient_booking.get('doctor', 'a doctor')} "
                    f"for {patient_booking.get('date', 'a date not listed')} at "
                    f"{patient_booking.get('time', 'a time not listed')}. Status: "
                    f"{patient_booking.get('booking_status', 'Confirmed')}. Contact the clinic if these details need changing."
                )
            return (
                "I couldn't find an appointment linked to your signed-in account. You can request one "
                "on the Booking page; if you booked another way, please contact the clinic."
            )
        if words & {"when", "time", "slot", "slots", "available", "availability"}:
            suggestions = get_smart_slot_recommendations()
            if suggestions:
                listed_slots = "; ".join(
                    f"{item['doctor']} at {item['time']}" for item in suggestions[:4]
                )
                return (
                    f"These are the next demo slots currently shown for {suggestions[0]['date']}: "
                    f"{listed_slots}. Open the Booking page to choose a doctor and confirm an appointment."
                )
        return (
            "To request an appointment, open Booking, enter your name, choose a department and doctor, "
            "select a date and time, then confirm. A booking confirmation will appear if the request is accepted."
        )

    if words & {"department", "departments", "service", "services", "specialty", "specialties"}:
        departments = ", ".join(
            specialty for specialty in DOCTOR_SPECIALTIES
            if get_doctors_for_specialty(specialty)
        )
        return (
            f"Departments with doctors listed in this demo are: {departments or 'none yet'}. "
            "Booking filters the doctor list by the selected department. Choose the closest specialty "
            "or contact the clinic if you are unsure which service is right."
        )

    if words & {"queue", "wait", "waiting", "eta", "line", "turn"}:
        live_queue = build_live_queue(queue_data)
        mean_wait = round(
            sum(patient["estimated_wait"] for patient in live_queue) / len(live_queue)
        ) if live_queue else 0
        return (
            f"The current sample queue has {len(live_queue)} patients and an estimated average wait "
            f"of about {mean_wait} minutes. Individual estimates can change; check the Queue Dashboard "
            "or confirm with clinic staff."
        )

    if words & {"portal", "account", "register", "registration", "password", "login", "signin", "sign"}:
        if words & {"reset", "forgot", "forgotten", "change"} and "password" in words:
            return (
                "Password reset isn't available in this demo yet. Please contact the clinic or system "
                "administrator; don't share your password in chat."
            )
        return (
            "Use Register to create a patient account with your email and a password of at least "
            "8 characters. Sign in later with the same email and password, then open Patient Portal."
        )

    if words & {"lab", "labs", "test", "tests", "result", "results", "report", "reports"}:
        return (
            "For test or lab results, sign in and check Patient Portal if your clinic has added them. "
            "This demo does not store real lab reports; contact your care team to interpret results."
        )

    if words & {"hours", "open", "opening", "close", "closing", "location", "address", "where", "phone", "contact"}:
        return (
            "Clinic hours, address, and reception contact details haven't been configured in this demo. "
            "Please use the contact details provided directly by your clinic."
        )

    if words & {"privacy", "private", "secure", "security", "data"}:
        return (
            "This is a demonstration application using sample data and basic accounts, not a certified "
            "clinical record system. Avoid entering sensitive health information here. For your clinic's "
            "privacy practices, contact the clinic directly."
        )

    if words & {"reminder", "reminders", "prescription", "prescriptions", "record", "records", "refill"}:
        return (
            "Sign in and open Patient Portal to see the sample care reminders and patient tools. "
            "This demo does not provide real prescriptions or medical records; verify all care instructions with your clinician."
        )

    return (
        "I couldn't confidently match that question to the information configured for this clinic. "
        "I can help with appointments, available doctors, queue estimates, registration, and portal navigation. "
        "For clinic-specific information, please ask reception; for medical advice, contact a licensed clinician."
    )


@app.get("/api/patient-convenience")
def patient_convenience_api():
    return jsonify({"features": patient_convenience_features})


@app.get("/api/booking-insights")
def booking_insights_api():
    live_queue = build_live_queue(queue_data)
    specialty = request.args.get("department")
    appointment_date = request.args.get("date")
    if appointment_date:
        try:
            datetime.strptime(appointment_date, "%Y-%m-%d")
        except ValueError:
            return jsonify({"success": False, "message": "Enter the appointment date as YYYY-MM-DD."}), 400
    return jsonify(
        {
            "summary": build_queue_summary(live_queue),
            "recommendations": get_smart_slot_recommendations(specialty, appointment_date),
            "warnings": [
                "High priority case load is above target for this morning.",
                "Manual follow-up is recommended for patients waiting over 20 minutes.",
            ],
        }
    )


@app.get("/api/clinical-overview")
def clinical_overview_api():
    return jsonify(build_health_dashboard())


@app.post("/api/doctor-login")
def doctor_login_api():
    payload = request.get_json(silent=True) or {}
    username = (payload.get("username") or "").strip().lower()
    password = payload.get("password") or ""
    account = next((doctor for doctor in doctor_accounts if doctor.get("username") == username), None)
    if not account or not check_password_hash(account.get("password_hash", ""), password):
        return jsonify({"success": False, "message": "Invalid doctor login details."}), 401

    session.clear()
    session["doctor_name"] = account["name"]
    session["doctor_username"] = username
    return jsonify({"success": True, "doctor": account["name"], "redirect": "/doctor-panel"})


@app.post("/api/doctor-register")
def doctor_register_api():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    username = (payload.get("username") or "").strip().lower()
    password = payload.get("password") or ""
    specialty = (payload.get("specialty") or "").strip()

    if not name or not username or not password or not specialty:
        return jsonify({"success": False, "message": "Doctor name, username, specialty, and password are required."}), 400
    if len(name) > 100 or len(username) > 40 or len(password) < 8:
        return jsonify({"success": False, "message": "Use a name up to 100 characters, username up to 40 characters, and password with at least 8 characters."}), 400
    if specialty not in DOCTOR_SPECIALTIES:
        return jsonify({"success": False, "message": "Choose a listed medical specialty."}), 400
    if not all(character.isalnum() or character in "._-" for character in username):
        return jsonify({"success": False, "message": "Username can contain letters, numbers, dots, underscores, and hyphens."}), 400
    if any(doctor.get("username", "").lower() == username for doctor in doctor_accounts):
        return jsonify({"success": False, "message": "That username is already registered."}), 409

    account = {
        "name": name if name.lower().startswith("dr.") else f"Dr. {name}",
        "username": username,
        "specialty": specialty,
        "password_hash": generate_password_hash(password),
    }
    doctor_accounts.append(account)
    save_doctors(doctor_accounts)
    session.clear()
    session["doctor_name"] = account["name"]
    session["doctor_username"] = username
    return jsonify({"success": True, "doctor": account["name"], "redirect": "/doctor-panel"}), 201


@app.get("/api/doctors")
def doctors_api():
    specialty = request.args.get("department", "").strip()
    directory = get_doctors_for_specialty(specialty) if specialty else doctor_accounts
    return jsonify({
        "doctors": [
            {"name": doctor["name"], "username": doctor["username"], "specialty": doctor.get("specialty", "Unspecified")}
            for doctor in directory
        ]
    })


@app.post("/api/doctor-specialty")
def update_doctor_specialty_api():
    username = session.get("doctor_username")
    account = next((doctor for doctor in doctor_accounts if doctor.get("username") == username), None)
    if not account:
        return jsonify({"success": False, "message": "Sign in as a doctor before updating your specialty."}), 401

    payload = request.get_json(silent=True) or {}
    specialty = (payload.get("specialty") or "").strip()
    if specialty not in DOCTOR_SPECIALTIES:
        return jsonify({"success": False, "message": "Choose a listed medical specialty."}), 400

    account["specialty"] = specialty
    save_doctors(doctor_accounts)
    return jsonify({"success": True, "doctor": account["name"], "specialty": specialty})


@app.post("/api/doctor-logout")
def doctor_logout_api():
    session.clear()
    return jsonify({"success": True})


@app.post("/api/patient-assignment")
def patient_assignment_api():
    payload = request.get_json(silent=True) or {}
    patient_name = (payload.get("patient_name") or "").strip()
    doctor_name = (payload.get("doctor_name") or "").strip()
    room_name = (payload.get("room_name") or "Room 1").strip()
    status = (payload.get("status") or "Assigned").strip()
    priority = (payload.get("priority") or "routine").strip().lower()

    if not patient_name or not doctor_name:
        return jsonify({"success": False, "message": "Patient and doctor are required."}), 400

    existing = next((item for item in assigned_patients if item["patient"] == patient_name), None)
    if existing:
        existing.update({"doctor": doctor_name, "room": room_name, "status": status, "priority": priority})
    else:
        assigned_patients.insert(0, {"patient": patient_name, "doctor": doctor_name, "status": status, "room": room_name, "priority": priority})

    return jsonify({"success": True, "assignment": {"patient": patient_name, "doctor": doctor_name, "room": room_name, "status": status}})


@app.get("/api/patient-assignments")
def patient_assignments_api():
    return jsonify({"assignments": assigned_patients})


@app.get("/api/reminders")
def reminders_api():
    return jsonify({"reminders": care_reminders})


@app.post("/api/reminders")
def send_reminder_api():
    payload = request.get_json(silent=True) or {}
    patient_name = (payload.get("patient_name") or "").strip()
    channel = (payload.get("channel") or "SMS").strip().lower()
    if not patient_name:
        return jsonify({"success": False, "message": "Patient name is required."}), 400

    reminder = {
        "patient": patient_name,
        "type": "Reminder",
        "channel": channel,
        "message": f"{channel.upper()} reminder sent to {patient_name} for their next scheduled consultation.",
        "due": "Now",
    }
    care_reminders.insert(0, reminder)
    return jsonify({"success": True, "reminder": reminder})


@app.get("/export/report.csv")
def export_report_csv():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["patient", "doctor", "status", "priority", "room"])
    for item in assigned_patients:
        writer.writerow([item["patient"], item["doctor"], item["status"], item["priority"], item["room"]])

    response = app.response_class(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=carequeue_report.csv"
    return response


@app.get("/report")
def hospital_report():
    live_queue = build_live_queue(queue_data)
    return render_template(
        "report.html",
        summary=build_queue_summary(live_queue),
        assignments=assigned_patients,
        alerts=notifications,
        now=datetime.today().strftime("%d %b %Y, %H:%M"),
    )


@app.get("/api/patient-search")
def patient_search_api():
    patient_name = request.args.get("name", "").strip()
    if not patient_name:
        return jsonify({"success": False, "message": "Patient name is required."}), 400

    record = lookup_patient_record(patient_name)
    if not record:
        return jsonify({"success": False, "message": "No patient record found."}), 404

    return jsonify({"success": True, "record": record})


@app.post("/api/patient-search")
def patient_search_post_api():
    payload = request.get_json(silent=True) or {}
    patient_name = (payload.get("name") or "").strip()
    if not patient_name:
        return jsonify({"success": False, "message": "Patient name is required."}), 400

    record = lookup_patient_record(patient_name)
    if not record:
        return jsonify({"success": False, "message": "No patient record found."}), 404

    return jsonify({"success": True, "record": record})


@app.post("/api/triage")
def triage_api():
    payload = request.get_json(silent=True) or {}
    symptoms = (payload.get("symptoms") or "").lower()
    risk = "routine"
    if any(term in symptoms for term in ["chest", "breath", "severe", "bleeding", "stroke", "unconscious"]):
        risk = "emergency"
    elif any(term in symptoms for term in ["fever", "pain", "dizziness", "vomit"]):
        risk = "urgent"
    recommended_wait = 10 if risk == "routine" else 5 if risk == "urgent" else 2
    return jsonify(
        {
            "risk": risk,
            "recommended_wait": recommended_wait,
            "recommended_action": "Immediate clinical review" if risk == "emergency" else "Prioritized queue placement" if risk == "urgent" else "Routine consultation",
        }
    )


@app.get("/api/bookings")
def bookings_api():
    return jsonify({"bookings": bookings})


@app.post("/api/bookings")
def create_booking():
    payload = request.get_json(silent=True) or {}
    if not payload:
        return jsonify({"success": False, "message": "No booking data supplied"}), 400

    valid, message = validate_booking(payload)
    if not valid:
        return jsonify({"success": False, "message": message}), 409

    priority = normalize_priority(payload.get("priority"))
    patient_name = (payload.get("patient_name") or payload.get("name") or "New Patient").strip()
    doctor_name = (payload.get("doctor") or "Dr. Aisha Patel").strip()
    date_value = (payload.get("date") or datetime.today().strftime("%Y-%m-%d")).strip()
    time_value = (payload.get("time") or get_next_available_slot(doctor_name, date_value)).strip()

    new_booking = {
        "name": patient_name,
        "department": payload.get("department") or "General Medicine",
        "doctor": doctor_name,
        "date": date_value,
        "time": time_value,
        "priority": priority,
        "email": payload.get("email") or "unknown@example.com",
        "phone": payload.get("phone") or "0000000000",
        "risk_level": "high" if priority == "emergency" else "medium" if priority == "urgent" else "low",
        "booking_status": "Confirmed",
    }
    bookings.insert(0, new_booking)
    save_bookings(bookings)

    return jsonify({
        "success": True,
        "booking": new_booking,
        "recommended_slot": get_next_available_slot(doctor_name, date_value),
        "message": "Appointment scheduled successfully with queue-aware safeguards in place.",
    }), 201


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
