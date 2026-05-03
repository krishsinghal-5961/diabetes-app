# DiabetesIQ — Deployment Guide

## Project Structure
```
diabetes_app/
├── app.py
├── requirements.txt
├── Procfile
├── templates/
│   └── index.html
└── models/
    ├── nb_model.pkl
    ├── mlp_model.pkl
    ├── scaler.pkl
    └── risk_ctrl.pkl
```

## Step 1 — Add your models
Unzip the models.zip you downloaded from Colab and paste the 4 .pkl files into the models/ folder.

## Step 2 — Test locally
```bash
pip install -r requirements.txt
pip install gunicorn
python app.py
```
Open http://localhost:5000

## Step 3 — Deploy on Render
1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
5. Click Deploy — done, you get a live URL in ~2 minutes
