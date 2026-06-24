from flask import Flask, render_template, request, redirect, url_for, session
import os
import io
import base64
import sqlite3
from datetime import timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import joblib
import librosa
import librosa.display
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from flask import send_file

# ---------------- APP SETUP ---------------- #
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_key")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=10)

# Global Constants for Audio Processing
SR = 22050
N_MFCC = 13

@app.before_request
def make_session_permanent():
    session.permanent = True

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'temp_uploads')
ALLOWED_EXTENSIONS = {'wav'}

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- DATABASE ---------------- #
def get_db_connection():
    db_path = os.path.join(os.getcwd(), 'database.db')

    conn = sqlite3.connect(
        db_path,
        timeout=30,
        check_same_thread=False
    )

    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        role TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL)''')

    conn.execute('''CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        patient_name TEXT NOT NULL,
        patient_age INTEGER NOT NULL,
        patient_gender TEXT NOT NULL,
        model_used TEXT NOT NULL,
        result TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()

init_db()

# ---------------- LOAD MODELS ---------------- #
rf_model, cnn_model = None, None

try:
    # Updated filenames to match your generated models
    rf_model = joblib.load("murmur_model.joblib")
    cnn_model = tf.keras.models.load_model("cnn_model/cnn_best_model1.keras")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"⚠️ Model loading error: {e}")

# ---------------- HELPER FUNCTIONS FOR RF FEATURE EXTRACTION ---------------- #

def extract_features(audio, sr=SR, n_mfcc=N_MFCC):
    """Extract handcrafted audio features matching the training notebook."""
    f = {}
    f['rms_mean']       = float(np.mean(librosa.feature.rms(y=audio)))
    f['zcr_mean']       = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    f['centroid_mean']  = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    f['bandwidth_mean'] = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    f['energy']         = float(np.sum(audio ** 2))
    f['skew']           = float(skew(audio))
    f['kurtosis']       = float(kurtosis(audio))

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        f[f'mfcc_{i+1}'] = float(np.mean(mfccs[i]))

    return f

# ---------------- ML FUNCTIONS ---------------- #
def predict_rf(av_path, mv_path, pv_path, tv_path):
    if rf_model is None:
        return "❌ RF model not loaded"

    feature_vector = {}

    for region, path in zip(["AV", "MV", "PV", "TV"], [av_path, mv_path, pv_path, tv_path]):
        if path is None:
            continue
        try:
            audio, sr = librosa.load(path, sr=SR)
            features = extract_features(audio, sr=SR, n_mfcc=N_MFCC) 
            for key, value in features.items():
                feature_vector[f"{region}_{key}"] = value
        except Exception as e:
            return f"Error processing {region}: {str(e)}"

    if len(feature_vector) == 0:
        return "Please upload at least one audio file."

    try:
        X = pd.DataFrame([feature_vector]).fillna(0)
        # Reindexing to ensure the features match the training order
        probs = rf_model.predict_proba(X)[0]
        # Using 0.3 threshold as per your notebook logic
        prediction = 1 if probs[1] >= 0.3 else 0
        confidence = probs[prediction] * 100

        if prediction == 1:
            return f"🔴 Murmur Detected (RF)<br>Confidence: {confidence:.2f}%"
        else:
            return f"🟢 No Murmur Detected (RF)<br>Confidence: {confidence:.2f}%"
    except Exception as e:
        return f"Model Prediction Error: {e}"

def predict_cnn(audio_path):
    if cnn_model is None:
        return "❌ CNN model not loaded"

    if audio_path is None:
        return "Please upload an audio file."

    try:
        audio, sr = librosa.load(audio_path, sr=SR)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if mel_db.shape[1] < 128:
            mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])))

        mel_db = mel_db[:, :128]
        # Normalization
        dmin, dmax = mel_db.min(), mel_db.max()
        if dmax - dmin > 0:
            mel_db = (mel_db - dmin) / (dmax - dmin)
            
        X = mel_db.reshape(1, 64, 128, 1)

        probs = cnn_model.predict(X, verbose=0)[0]
        prediction = np.argmax(probs)
        confidence = probs[prediction] * 100

        if prediction == 1:
            return f"🔴 Murmur Detected (CNN)<br>Confidence: {confidence:.2f}%"
        else:
            return f"🟢 No Murmur Detected (CNN)<br>Confidence: {confidence:.2f}%"
    except Exception as e:
        return f"Error: {str(e)}"

def get_spectrogram_base64(audio_path, title):
    if not audio_path or not os.path.exists(audio_path):
        return None

    audio, sr = librosa.load(audio_path, sr=SR)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(5, 3))
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(f"Spectrogram: {title}")

    plt.tight_layout()
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)

    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

# ---------------- ROUTES ---------------- #
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session["logged_in"] = True
            session["user"] = username
            session["full_name"] = user["full_name"]
            return redirect(url_for("home"))
        else:
            error = "❌ Invalid username or password."
    return render_template("login.html", error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None

    if request.method == "POST":

        full_name = request.form.get("full_name")
        role = request.form.get("role")
        email = request.form.get("email")
        username = request.form.get("username")
        password = request.form.get("password")

        try:
            conn = get_db_connection()

            existing = conn.execute(
                "SELECT * FROM users WHERE username=? OR email=?",
                (username, email)
            ).fetchone()

            if existing:
                error = "❌ Username or Email already exists."

            else:
                conn.execute(
                    """
                    INSERT INTO users
                    (full_name, role, email, username, password)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        full_name,
                        role,
                        email,
                        username,
                        generate_password_hash(password)
                    )
                )

                conn.commit()

                return redirect(url_for("login"))

        except Exception as e:
            error = str(e)

        finally:
            conn.close()

    return render_template("register.html", error=error)

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    username = session["user"]
    result = None
    spectrograms = {}
    history = []  # To store history records

    if request.method == "POST":
        model_type = request.form.get("model_type")
        # Get patient details from the form
        p_name = request.form.get("patient_name", "Unknown")
        p_age = request.form.get("patient_age", 0)
        p_gender = request.form.get("patient_gender", "Unknown")

        file_paths = {"AV": None, "MV": None, "PV": None, "TV": None}

        for region in file_paths:
            file = request.files.get(f"{region}_audio")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(UPLOAD_FOLDER, f"{region}_{filename}")
                file.save(path)
                file_paths[region] = path

        # Perform Prediction
        if model_type == "rf":
            result = predict_rf(*file_paths.values())
        elif model_type == "cnn":
            audio = next((v for v in file_paths.values() if v), None)
            result = predict_cnn(audio)

        # --- SAVE TO DATABASE HISTORY ---
        if result:
            try:
                conn = get_db_connection()
                conn.execute('''INSERT INTO history 
                    (username, patient_name, patient_age, patient_gender, model_used, result) 
                    VALUES (?, ?, ?, ?, ?, ?)''', 
                    (username, p_name, p_age, p_gender, model_type.upper(), result))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"❌ Database Save Error: {e}")

        # Generate Spectrograms for UI
        for region, path in file_paths.items():
            if path:
                spectrograms[region] = get_spectrogram_base64(path, region)

        # Cleanup files
        for path in file_paths.values():
            if path and os.path.exists(path):
                os.remove(path)

    # --- RETRIEVE HISTORY FOR THE USER ---
    try:
        conn = get_db_connection()
        history = conn.execute('SELECT * FROM history WHERE username = ? ORDER BY timestamp DESC', 
                               (username,)).fetchall()
        conn.close()
    except Exception as e:
        print(f"❌ Database Retrieval Error: {e}")

    return render_template(
    "dashboard.html",
    user=username,
    full_name=session.get("full_name"),
    result=result,
    spectrograms=spectrograms,
    history=history
)
    
    
@app.route("/download_report")
def download_report():

    patient_name = request.args.get("patient_name", "Unknown")
    patient_age = request.args.get("patient_age", "N/A")
    patient_gender = request.args.get("patient_gender", "N/A")
    result = request.args.get("result", "No Result")

    pdf_path = "Heart_Murmur_Report.pdf"

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph("Heart Murmur Detection Report", styles["Title"])
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(f"<b>Patient Name:</b> {patient_name}",
        styles["Normal"])
    )

    elements.append(
        Paragraph(f"<b>Age:</b> {patient_age}",
        styles["Normal"])
    )

    elements.append(
        Paragraph(f"<b>Gender:</b> {patient_gender}",
        styles["Normal"])
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            f"<b>Diagnostic Result:</b> {result}",
            styles["Heading2"]
        )
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "This report was generated automatically "
            "using Machine Learning analysis.",
            styles["Normal"]
        )
    )

    doc.build(elements)

    return send_file(
        pdf_path,
        as_attachment=True
    )
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/analytics")
def analytics():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template("analytics.html")

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)