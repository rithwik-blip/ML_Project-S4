"""
main.py — FastAPI backend for Match Outcome Prediction
Serves predictions from trained RF + ANN models, dataset analytics,
model metrics, and prediction history.
"""

import os, json, datetime
from typing import Optional
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Try TensorFlow
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ---------------------------------------------------------------------------
# Load models & metadata
# ---------------------------------------------------------------------------
def load_assets():
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        raise RuntimeError(
            "Models not trained yet! Run: python train_models.py")

    with open(meta_path) as f:
        metadata = json.load(f)

    rf_fb = joblib.load(os.path.join(MODEL_DIR, "rf_football.joblib"))
    rf_ck = joblib.load(os.path.join(MODEL_DIR, "rf_cricket.joblib"))
    scaler_fb = joblib.load(os.path.join(MODEL_DIR, "scaler_football.joblib"))
    scaler_ck = joblib.load(os.path.join(MODEL_DIR, "scaler_cricket.joblib"))
    le_fb = joblib.load(os.path.join(MODEL_DIR, "le_football.joblib"))
    le_team1 = joblib.load(os.path.join(MODEL_DIR, "le_cricket_team1.joblib"))
    le_team2 = joblib.load(os.path.join(MODEL_DIR, "le_cricket_team2.joblib"))

    ann_fb = None
    ann_ck = None
    if HAS_TF:
        ann_fb_path = os.path.join(MODEL_DIR, "ann_football.keras")
        ann_ck_path = os.path.join(MODEL_DIR, "ann_cricket.keras")
        if os.path.exists(ann_fb_path):
            ann_fb = tf.keras.models.load_model(ann_fb_path)
        if os.path.exists(ann_ck_path):
            ann_ck = tf.keras.models.load_model(ann_ck_path)

    cricket_match_df = pd.read_csv(
        os.path.join(MODEL_DIR, "cricket_match_data.csv"))

    return {
        "metadata": metadata,
        "rf_fb": rf_fb, "rf_ck": rf_ck,
        "ann_fb": ann_fb, "ann_ck": ann_ck,
        "scaler_fb": scaler_fb, "scaler_ck": scaler_ck,
        "le_fb": le_fb, "le_team1": le_team1, "le_team2": le_team2,
        "cricket_match_df": cricket_match_df,
    }

assets = load_assets()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Match Outcome Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory prediction history
prediction_history = []

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class FootballPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    b365h: Optional[float] = None
    b365d: Optional[float] = None
    b365a: Optional[float] = None

class CricketPredictionRequest(BaseModel):
    team1: str  # batting first
    team2: str  # chasing

from fastapi.responses import FileResponse

FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Match Outcome Predictor API", "status": "running"}

@app.get("/style.css")
def get_css():
    css_file = os.path.join(FRONTEND_DIR, "style.css")
    if os.path.exists(css_file):
        return FileResponse(css_file, media_type="text/css")
    raise HTTPException(404, "style.css not found")

@app.get("/app.js")
def get_js():
    js_file = os.path.join(FRONTEND_DIR, "app.js")
    if os.path.exists(js_file):
        return FileResponse(js_file, media_type="application/javascript")
    raise HTTPException(404, "app.js not found")

@app.get("/api/health")
def health():
    return {"status": "ok", "has_tensorflow": HAS_TF,
            "models_loaded": assets["rf_fb"] is not None}

# --- Dataset stats ---
@app.get("/api/stats")
def get_stats():
    meta = assets["metadata"]
    return {
        "football": meta["football"]["dataset_stats"],
        "cricket": meta["cricket"]["dataset_stats"],
    }

# --- Football ---
@app.get("/api/football/teams")
def football_teams():
    return {"teams": assets["metadata"]["football"]["teams"]}

@app.get("/api/football/team-stats")
def football_team_stats():
    return assets["metadata"]["football"]["team_detailed"]

@app.post("/api/football/predict")
def predict_football(req: FootballPredictionRequest):
    meta = assets["metadata"]["football"]
    team_stats = meta["team_stats"]
    features = meta["features"]

    if req.home_team not in team_stats:
        raise HTTPException(404, f"Home team '{req.home_team}' not found")
    if req.away_team not in team_stats:
        raise HTTPException(404, f"Away team '{req.away_team}' not found")

    home = team_stats[req.home_team]
    away = team_stats[req.away_team]
    odds_avg = meta["odds_avg"]

    feat = {
        'HS': home.get('HS', 0), 'AS': away.get('AS', 0),
        'HST': home.get('HST', 0), 'AST': away.get('AST', 0),
        'HF': home.get('HF', 0), 'AF': away.get('AF', 0),
        'HC': home.get('HC', 0), 'AC': away.get('AC', 0),
        'HY': home.get('HY', 0), 'AY': away.get('AY', 0),
        'HR': home.get('HR', 0), 'AR': away.get('AR', 0),
        'B365H': req.b365h if req.b365h else odds_avg['B365H'],
        'B365D': req.b365d if req.b365d else odds_avg['B365D'],
        'B365A': req.b365a if req.b365a else odds_avg['B365A'],
    }

    X = pd.DataFrame([feat])[features]

    classes = meta["classes"]  # ['A', 'D', 'H']
    label_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}

    # RF prediction
    rf_probs = assets["rf_fb"].predict_proba(X)[0]
    rf_pred_idx = np.argmax(rf_probs)
    rf_pred = classes[rf_pred_idx]

    result = {
        "home_team": req.home_team,
        "away_team": req.away_team,
        "features_used": feat,
        "rf_prediction": label_map[rf_pred],
        "rf_probabilities": {label_map[c]: round(float(p), 4)
                             for c, p in zip(classes, rf_probs)},
    }

    # ANN prediction
    if assets["ann_fb"] is not None:
        X_s = assets["scaler_fb"].transform(X)
        ann_probs = assets["ann_fb"].predict(X_s, verbose=0)[0]
        ann_pred_idx = int(np.argmax(ann_probs))
        ann_pred = classes[ann_pred_idx]
        result["ann_prediction"] = label_map[ann_pred]
        result["ann_probabilities"] = {label_map[c]: round(float(p), 4)
                                        for c, p in zip(classes, ann_probs)}
    else:
        result["ann_prediction"] = "N/A (TensorFlow not available)"
        result["ann_probabilities"] = {}

    # Save to history
    prediction_history.append({
        "id": len(prediction_history) + 1,
        "sport": "Football",
        "teams": f"{req.home_team} vs {req.away_team}",
        "rf_prediction": result["rf_prediction"],
        "ann_prediction": result.get("ann_prediction", "N/A"),
        "rf_probabilities": result["rf_probabilities"],
        "ann_probabilities": result.get("ann_probabilities", {}),
        "timestamp": datetime.datetime.now().isoformat(),
    })

    return result

# --- Cricket ---
@app.get("/api/cricket/teams")
def cricket_teams():
    meta = assets["metadata"]["cricket"]
    return {
        "all_teams": meta["all_teams"],
        "teams_batting_first": meta["teams_batting_first"],
        "teams_chasing": meta["teams_chasing"],
    }

@app.get("/api/cricket/team-stats")
def cricket_team_stats():
    return assets["metadata"]["cricket"]["team_win_rates"]

@app.post("/api/cricket/predict")
def predict_cricket(req: CricketPredictionRequest):
    meta = assets["metadata"]["cricket"]
    bat_avgs = meta["batting_first_avgs"]
    chase_avgs = meta["chasing_avgs"]

    if req.team1 not in bat_avgs:
        raise HTTPException(404,
            f"Team '{req.team1}' has no batting-first history. "
            f"Available: {list(bat_avgs.keys())}")
    if req.team2 not in chase_avgs:
        raise HTTPException(404,
            f"Team '{req.team2}' has no chasing history. "
            f"Available: {list(chase_avgs.keys())}")

    le_team1 = assets["le_team1"]
    le_team2 = assets["le_team2"]

    def safe_encode(le, value):
        if value in le.classes_:
            return int(le.transform([value])[0])
        return 0

    t1_avg = bat_avgs[req.team1]
    t2_avg = chase_avgs[req.team2]

    feat = {
        'team1_enc': safe_encode(le_team1, req.team1),
        'team2_enc': safe_encode(le_team2, req.team2),
        'team1_runs': t1_avg['team1_runs'],
        'team1_wickets': t1_avg['team1_wickets'],
        'team1_extras': t1_avg['team1_extras'],
        'team1_fours': t1_avg['team1_fours'],
        'team1_sixes': t1_avg['team1_sixes'],
        'team1_run_rate': t1_avg['team1_run_rate'],
        'team2_extras': t2_avg['team2_extras'],
    }

    cricket_features = meta["features"]
    X = pd.DataFrame([feat])[cricket_features]

    # RF prediction
    rf_prob_team1 = float(assets["rf_ck"].predict_proba(X)[0][1])
    rf_pred = req.team1 if rf_prob_team1 >= 0.5 else req.team2

    result = {
        "team1": req.team1,
        "team2": req.team2,
        "features_used": feat,
        "rf_prediction": rf_pred,
        "rf_probabilities": {
            req.team1: round(rf_prob_team1, 4),
            req.team2: round(1 - rf_prob_team1, 4),
        },
    }

    # ANN prediction
    if assets["ann_ck"] is not None:
        X_s = assets["scaler_ck"].transform(X)
        ann_prob_team1 = float(assets["ann_ck"].predict(X_s, verbose=0)[0][0])
        ann_pred = req.team1 if ann_prob_team1 >= 0.5 else req.team2
        result["ann_prediction"] = ann_pred
        result["ann_probabilities"] = {
            req.team1: round(ann_prob_team1, 4),
            req.team2: round(1 - ann_prob_team1, 4),
        }
    else:
        result["ann_prediction"] = "N/A"
        result["ann_probabilities"] = {}

    prediction_history.append({
        "id": len(prediction_history) + 1,
        "sport": "Cricket",
        "teams": f"{req.team1} vs {req.team2}",
        "rf_prediction": result["rf_prediction"],
        "ann_prediction": result.get("ann_prediction", "N/A"),
        "rf_probabilities": result["rf_probabilities"],
        "ann_probabilities": result.get("ann_probabilities", {}),
        "timestamp": datetime.datetime.now().isoformat(),
    })

    return result

# --- Model Metrics ---
@app.get("/api/models/metrics")
def model_metrics():
    meta = assets["metadata"]
    return {
        "football": {
            "rf": meta["football"]["rf_metrics"],
            "ann": meta["football"]["ann_metrics"],
            "rf_confusion_matrix": meta["football"]["rf_confusion_matrix"],
            "ann_confusion_matrix": meta["football"]["ann_confusion_matrix"],
            "rf_feature_importance": meta["football"]["rf_feature_importance"],
            "classes": meta["football"]["classes"],
        },
        "cricket": {
            "rf": meta["cricket"]["rf_metrics"],
            "ann": meta["cricket"]["ann_metrics"],
            "rf_confusion_matrix": meta["cricket"]["rf_confusion_matrix"],
            "ann_confusion_matrix": meta["cricket"]["ann_confusion_matrix"],
            "rf_feature_importance": meta["cricket"]["rf_feature_importance"],
            "classes": meta["cricket"]["classes"],
        }
    }

# --- Analytics ---
@app.get("/api/analytics/football")
def football_analytics():
    meta = assets["metadata"]["football"]
    return {
        "dataset_stats": meta["dataset_stats"],
        "ftr_distribution": meta["ftr_distribution"],
        "team_detailed": meta["team_detailed"],
        "correlation": meta["correlation"],
        "rf_feature_importance": meta["rf_feature_importance"],
    }

@app.get("/api/analytics/cricket")
def cricket_analytics():
    meta = assets["metadata"]["cricket"]
    return {
        "dataset_stats": meta["dataset_stats"],
        "outcome_distribution": meta["outcome_distribution"],
        "top_batters": meta["top_batters"],
        "top_bowlers": meta["top_bowlers"],
        "runs_per_over": meta["runs_per_over"],
        "wickets_per_over": meta["wickets_per_over"],
        "team_runs": meta["team_runs"],
        "team_win_rates": meta["team_win_rates"],
        "ball_correlation": meta["ball_correlation"],
        "rf_feature_importance": meta["rf_feature_importance"],
    }

# --- Prediction History ---
@app.get("/api/history")
def get_history():
    return {"predictions": list(reversed(prediction_history))}

@app.delete("/api/history")
def clear_history():
    prediction_history.clear()
    return {"message": "History cleared"}
