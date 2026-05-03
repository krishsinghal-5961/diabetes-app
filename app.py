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

    fr = fuzzy_risk(features[1], features[5])
    features_with_fuzzy = features + [fr]

    scaled = scaler.transform([features_with_fuzzy])
    nb_result  = 'Diabetic' if nb.predict(scaled)[0] == 1 else 'Not Diabetic'
    mlp_result = 'Diabetic' if mlp.predict(scaled)[0] == 1 else 'Not Diabetic'
    nb_prob    = round(float(nb.predict_proba(scaled)[0][1]) * 100, 1)
    mlp_prob   = round(float(mlp.predict_proba(scaled)[0][1]) * 100, 1)

    return jsonify({
        'fuzzy_risk': fr,
        'nb_result':  nb_result,
        'nb_prob':    nb_prob,
        'mlp_result': mlp_result,
        'mlp_prob':   mlp_prob,
    })

if __name__ == '__main__':
    app.run(debug=True)
