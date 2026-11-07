from pathlib import Path
import csv
import io
import json
import sqlite3
from datetime import datetime, timezone
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response
from ml import ROOT, FEATURES, MODEL_PATH, METRICS_PATH, train

app = Flask(__name__)
DB = ROOT / 'soil_history.db'
LIMITS = {'pH': (0, 14), 'N': (0, 10000), 'P': (0, 10000), 'K': (0, 10000), 'OC': (0, 100), 'moisture': (0, 100)}

def database():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute('CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, pH REAL, N REAL, P REAL, K REAL, OC REAL, moisture REAL, health_class TEXT, confidence REAL)')
    conn.commit()
    return conn

def advice(label):
    return {'Healthy': 'Maintain soil monitoring and use crop-specific guidance for nutrient management.',
            'Moderate': 'Consider a laboratory soil test and review nutrient balance, pH and irrigation with an agronomist.',
            'Poor': 'Seek a laboratory soil test and local agronomy advice before adding fertilizers or amendments.'}.get(label, 'Confirm with a soil laboratory.')

def get_model():
    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)

@app.get('/')
def index():
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
    return render_template('index.html', metrics=metrics)

@app.post('/api/predict')
def predict():
    data = request.get_json(silent=True) or request.form
    try:
        values = {}
        for key in FEATURES:
            val = float(data[key])
            low, high = LIMITS[key]
            if not low <= val <= high or not __import__('math').isfinite(val):
                raise ValueError(f'{key} must be between {low} and {high}')
            values[key] = val
        model = get_model()
        frame = pd.DataFrame([values], columns=FEATURES)
        label = str(model.predict(frame)[0])
        probabilities = model.predict_proba(frame)[0]
        confidence = float(max(probabilities))
        conn = database()
        conn.execute('INSERT INTO predictions (created_at,pH,N,P,K,OC,moisture,health_class,confidence) VALUES (?,?,?,?,?,?,?,?,?)',
                     (datetime.now(timezone.utc).isoformat(), *[values[k] for k in FEATURES], label, confidence))
        conn.commit()
        conn.close()
        return jsonify({'health_class': label, 'confidence': round(confidence * 100, 1),
                        'advice': advice(label), 'values': values, 'demo': True})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({'error': str(exc)}), 400

@app.get('/api/history')
def history():
    conn = database()
    rows = [dict(row) for row in conn.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 100')]
    conn.close()
    return jsonify(rows)

@app.get('/api/report.csv')
def report():
    conn = database()
    rows = conn.execute('SELECT * FROM predictions ORDER BY id DESC').fetchall()
    conn.close()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['id', 'created_at', *FEATURES, 'health_class', 'confidence'])
    for row in rows:
        writer.writerow([row[col] for col in ['id', 'created_at', *FEATURES, 'health_class', 'confidence']])
    return Response(output.getvalue(), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=soil_health_report.csv'})

@app.get('/api/metrics')
def metrics():
    if not METRICS_PATH.exists():
        train()
    return jsonify(json.loads(METRICS_PATH.read_text()))

if __name__ == '__main__':
    database().close()
    app.run(debug=True, host='127.0.0.1', port=5000)
