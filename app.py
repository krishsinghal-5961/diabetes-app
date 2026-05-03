from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import numpy as np
import skfuzzy.control as ctrl

app = Flask(__name__)

nb     = joblib.load('models/nb_model.pkl')
mlp    = joblib.load('models/mlp_model.pkl')
scaler = joblib.load('models/scaler.pkl')

with open('models/risk_ctrl.pkl', 'rb') as f:
    risk_ctrl = pickle.load(f)

def fuzzy_risk(g, b):
    g = np.clip(g, 50, 200)
    b = np.clip(b, 15, 59)
    sim = ctrl.ControlSystemSimulation(risk_ctrl)
    sim.input['glucose'] = g
    sim.input['bmi'] = b
    sim.compute()
    return round(sim.output['risk'], 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # --- Input features ---
    features = [
        float(data['pregnancies']),
        float(data['glucose']),
        float(data['blood_pressure']),
        float(data['skin_thickness']),
        float(data['insulin']),
        float(data['bmi']),
        float(data['diabetes_pedigree']),
        float(data['age']),
    ]

    # --- Fuzzy logic ---
    fr = fuzzy_risk(features[1], features[5])  # glucose, bmi
    features_with_fuzzy = features + [fr]

    # --- Scaling ---
    scaled = scaler.transform([features_with_fuzzy])

    # --- Model Predictions ---
    nb_pred_val  = nb.predict(scaled)[0]
    mlp_pred_val = mlp.predict(scaled)[0]

    nb_prob_raw  = float(nb.predict_proba(scaled)[0][1])   # 0–1
    mlp_prob_raw = float(mlp.predict_proba(scaled)[0][1])  # 0–1

    # --- Calibration (important fix) ---
    nb_prob_raw = 0.7 * nb_prob_raw + 0.3 * 0.5

    # --- Labels ---
    nb_result  = 'Diabetic' if nb_pred_val == 1 else 'Not Diabetic'
    mlp_result = 'Diabetic' if mlp_pred_val == 1 else 'Not Diabetic'

    # --- Ensemble Logic ---
    fuzzy_prob = fr / 100  # convert to 0–1

    final_score = (nb_prob_raw + mlp_prob_raw + fuzzy_prob) / 3

    if final_score > 0.6:
        final_result = "Diabetic"
        risk_level = "High Risk"
    elif final_score > 0.3:
        final_result = "Moderate Risk"
        risk_level = "Moderate Risk"
    else:
        final_result = "Not Diabetic"
        risk_level = "Low Risk"

    # --- Convert to percentage for UI ---
    nb_prob  = round(nb_prob_raw * 100, 1)
    mlp_prob = round(mlp_prob_raw * 100, 1)
    final_score_percent = round(final_score * 100, 1)

    # --- Response ---
    return jsonify({
        'fuzzy_risk': round(fr, 2),

        'nb_result': nb_result,
        'nb_prob': nb_prob,

        'mlp_result': mlp_result,
        'mlp_prob': mlp_prob,

        'final_result': final_result,
        'final_score': final_score_percent,
        'risk_level': risk_level
    })

if __name__ == '__main__':
    app.run(debug=True)
