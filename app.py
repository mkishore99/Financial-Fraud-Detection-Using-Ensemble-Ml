from flask import Flask, render_template, request, redirect, url_for, session, send_file
import sqlite3
import pickle
import numpy as np
import pandas as pd
import os
import io

import matplotlib
matplotlib.use("Agg")          # ✅ MUST BE BEFORE pyplot
import matplotlib.pyplot as plt

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.metrics import classification_report





app = Flask(__name__)
app.secret_key = "supersecretkey"   # change in production

# ---------------- LOAD ML MODEL ----------------
model = pickle.load(open("fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
# ---------------- LOAD FEATURE NAMES ----------------
try:
    feature_names = pickle.load(open("features.pkl", "rb"))
except FileNotFoundError:
    # fallback: get feature names from scaler (sklearn >=1.0)
    feature_names = list(scaler.feature_names_in_)

# ---------------- DATABASE ----------------
def get_db():
    return sqlite3.connect("users.db")

# ---------------- AUTH ROUTES ----------------

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        try:
            db = get_db()
            cursor = db.cursor()
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password)
            )
            db.commit()
            db.close()
            return redirect(url_for("login"))
        except:
            return "Username already exists"

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        db.close()

        if user and check_password_hash(user[2], password):
            session["user"] = username
            return redirect(url_for("upload"))
        else:
            return "Invalid credentials"

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ---------------- HOME ----------------
@app.route("/")
def home():
    return redirect(url_for("login"))
    return redirect(url_for("upload"))

# ---------------- DATASET UPLOAD ----------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files.get("dataset")

        if not file or not file.filename.endswith(".csv"):
            return "Only CSV files allowed"

        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        data = pd.read_csv(file_path)

        # Remove unwanted columns safely
        for col in ["id", "ID"]:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)

        if "Class" not in data.columns:
            return "Dataset must contain 'Class' column"

        X = data.drop("Class", axis=1)
        y_true = data["Class"]

        # Ensure correct feature order
        X = X[scaler.feature_names_in_]

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        total = len(data)
        fraud = int(y_pred.sum())
        legit = total - fraud
        fraud_percent = round((fraud / total) * 100, 2)

        report = classification_report(y_true, y_pred)

        # Store for PDF
        session["report_data"] = {
            "total": total,
            "fraud": fraud,
            "legit": legit,
            "fraud_percent": fraud_percent,
            "classification": report
        }
        # ---------- CHART ----------
        os.makedirs("static/charts", exist_ok=True)

        labels = ["Legitimate", "Fraud"]
        values = [legit, fraud]

        plt.figure()
        plt.bar(labels, values)
        plt.title("Transaction Classification")
        plt.xlabel("Type")
        plt.ylabel("Count")

        chart_path = "static/charts/result.png"
        plt.savefig(chart_path)
        plt.close()

        return render_template(
            "report.html",
            total=total,
            fraud=fraud,
            legit=legit,
            fraud_percent=fraud_percent,
            report=report,
            chart="charts/result.png"
        )
    return render_template("upload.html")

@app.route("/download_report")
def download_report():
    if "user" not in session:
        return redirect(url_for("login"))

    report_data = session.get("report_data")
    if not report_data:
        return "No report available"

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, height - 50, "Financial Fraud Detection Report")

    pdf.setFont("Helvetica", 11)
    y = height - 100

    pdf.drawString(50, y, f"Total Transactions: {report_data['total']}")
    y -= 20
    pdf.drawString(50, y, f"Fraud Transactions: {report_data['fraud']}")
    y -= 20
    pdf.drawString(50, y, f"Legitimate Transactions: {report_data['legit']}")
    y -= 20
    pdf.drawString(50, y, f"Fraud Percentage: {report_data['fraud_percent']}%")
    y -= 40

    pdf.drawString(50, y, "Classification Report:")
    y -= 20

    for line in report_data["classification"].split("\n"):
        pdf.drawString(50, y, line)
        y -= 15
        if y < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y = height - 50

    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="fraud_detection_report.pdf",
        mimetype="application/pdf"
    )

@app.route("/manual", methods=["GET", "POST"])
def manual():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    probability = None

    if request.method == "POST":
        values = []
        for col in feature_names:
            values.append(float(request.form[col]))

        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        prediction = "🚨 Fraudulent Transaction" if pred == 1 else "✅ Legitimate Transaction"
        probability = round(prob * 100, 2)

    return render_template(
        "manual.html",
        feature_names=feature_names,
        prediction=prediction,
        probability=probability
    )

@app.route("/download_pdf")
def download_pdf():
    if "user" not in session or "report_data" not in session:
        return redirect(url_for("login"))

    data = session["report_data"]

    os.makedirs("reports", exist_ok=True)
    file_path = "reports/fraud_report.pdf"

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Financial Fraud Detection Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"<b>User:</b> {session['user']}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    table_data = [
        ["Total Transactions", data["total"]],
        ["Fraudulent Transactions", data["fraud"]],
        ["Legitimate Transactions", data["legit"]],
        ["Fraud Percentage", f"{data['fraud_percent']}%"]
    ]

    table = Table(table_data)
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("<b>Classification Report</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    for line in data["classification"].split("\n"):
        elements.append(Paragraph(line.replace(" ", "&nbsp;"), styles["Normal"]))

    doc.build(elements)

    return send_file(file_path, as_attachment=True)

# ---------------- SINGLE TRANSACTION PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    features = [float(x) for x in request.form.values()]
    features = scaler.transform([features])
    feature_names = pickle.load(open("features.pkl", "rb"))

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    # Load feature names
    try:
        feature_names = pickle.load(open("features.pkl", "rb"))
    except:
        # Fallback if features.pkl is missing
        feature_names = scaler.feature_names_in_.tolist()

    result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"

    return render_template(
        "result.html",
        prediction=result,
        probability=round(probability * 100, 2)
    )

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
