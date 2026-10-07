"""
train_models.py
===============
Trains all four ML models (RF + ANN for Cricket and Football) using the exact
pipeline from the original Jupyter notebook, then serialises everything the
FastAPI backend needs to serve predictions and analytics.

Run once:  python train_models.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Try to import TensorFlow
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
    print("TensorFlow version:", tf.__version__)
except ImportError:
    HAS_TF = False
    print("WARNING: TensorFlow not found. Will train only Random Forest models.")
    print("Install with: pip install tensorflow")

# ===========================================================================
# PART 1 — FOOTBALL
# ===========================================================================
print("\n" + "="*60)
print("PART 1: FOOTBALL (EPL 2018-19)")
print("="*60)

football_df = pd.read_csv(os.path.join(DATA_DIR, "E0_2018-2019.csv"),
                           encoding="latin1")
print(f"Loaded football data: {football_df.shape}")

football_features = ['HS','AS','HST','AST','HF','AF','HC','AC',
                      'HY','AY','HR','AR','B365H','B365D','B365A']

football_df = football_df.dropna(subset=football_features + ['FTR']).reset_index(drop=True)

X_fb = football_df[football_features].copy()

le_fb = LabelEncoder()
y_fb = le_fb.fit_transform(football_df['FTR'])  # A=0, D=1, H=2
print("Classes:", list(le_fb.classes_))

X_fb_train, X_fb_test, y_fb_train, y_fb_test = train_test_split(
    X_fb, y_fb, test_size=0.2, random_state=SEED, stratify=y_fb
)

scaler_fb = StandardScaler()
X_fb_train_s = scaler_fb.fit_transform(X_fb_train)
X_fb_test_s = scaler_fb.transform(X_fb_test)

print(f"Train: {X_fb_train.shape}  Test: {X_fb_test.shape}")

# --- Random Forest (Football) ---
rf_fb = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=SEED)
rf_fb.fit(X_fb_train, y_fb_train)
rf_fb_pred = rf_fb.predict(X_fb_test)

rf_fb_metrics = {
    "accuracy": round(accuracy_score(y_fb_test, rf_fb_pred), 4),
    "precision": round(precision_score(y_fb_test, rf_fb_pred, average='macro', zero_division=0), 4),
    "recall": round(recall_score(y_fb_test, rf_fb_pred, average='macro', zero_division=0), 4),
    "f1_score": round(f1_score(y_fb_test, rf_fb_pred, average='macro', zero_division=0), 4),
}
rf_fb_cm = confusion_matrix(y_fb_test, rf_fb_pred).tolist()
rf_fb_importance = {feat: round(imp, 4) for feat, imp in
                     zip(football_features, rf_fb.feature_importances_)}

print(f"RF Football — Acc: {rf_fb_metrics['accuracy']}")

# --- ANN (Football) ---
ann_fb_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
ann_fb_cm = [[0]*3]*3

if HAS_TF:
    ann_fb = keras.Sequential([
        layers.Input(shape=(X_fb_train_s.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(le_fb.classes_), activation='softmax')
    ])
    ann_fb.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True)

    ann_fb.fit(X_fb_train_s, y_fb_train, validation_split=0.15,
               epochs=150, batch_size=16, callbacks=[early_stop], verbose=0)

    ann_fb_probs = ann_fb.predict(X_fb_test_s, verbose=0)
    ann_fb_pred = np.argmax(ann_fb_probs, axis=1)

    ann_fb_metrics = {
        "accuracy": round(accuracy_score(y_fb_test, ann_fb_pred), 4),
        "precision": round(precision_score(y_fb_test, ann_fb_pred, average='macro', zero_division=0), 4),
        "recall": round(recall_score(y_fb_test, ann_fb_pred, average='macro', zero_division=0), 4),
        "f1_score": round(f1_score(y_fb_test, ann_fb_pred, average='macro', zero_division=0), 4),
    }
    ann_fb_cm = confusion_matrix(y_fb_test, ann_fb_pred).tolist()
    ann_fb.save(os.path.join(MODEL_DIR, "ann_football.keras"))
    print(f"ANN Football — Acc: {ann_fb_metrics['accuracy']}")
else:
    print("Skipping ANN Football (no TensorFlow)")

# ===========================================================================
# PART 2 — CRICKET
# ===========================================================================
print("\n" + "="*60)
print("PART 2: CRICKET (IPL)")
print("="*60)

cricket_raw = pd.read_csv(os.path.join(DATA_DIR, "deliveries.csv"))
print(f"Loaded cricket data: {cricket_raw.shape}")

cricket_raw = cricket_raw[cricket_raw['inning'].isin([1, 2])].copy()

def aggregate_innings(group):
    runs = group['total_runs'].sum()
    wickets = group['is_wicket'].sum()
    extras = group['extra_runs'].sum()
    balls = len(group)
    fours = (group['batsman_runs'] == 4).sum()
    sixes = (group['batsman_runs'] == 6).sum()
    overs = balls / 6 if balls > 0 else 0
    run_rate = runs / overs if overs > 0 else 0
    return pd.Series({
        'team': group['batting_team'].iloc[0],
        'opponent': group['bowling_team'].iloc[0],
        'runs': runs, 'wickets': wickets, 'extras': extras,
        'balls': balls, 'fours': fours, 'sixes': sixes, 'run_rate': run_rate
    })

innings_df = (cricket_raw.groupby(['match_id', 'inning'])
              .apply(aggregate_innings)
              .reset_index())

records = []
for match_id, g in innings_df.groupby('match_id'):
    g = g.set_index('inning')
    if 1 not in g.index or 2 not in g.index:
        continue
    inn1, inn2 = g.loc[1], g.loc[2]
    team1_won = 1 if inn1['runs'] > inn2['runs'] else 0
    records.append({
        'match_id': match_id,
        'team1': inn1['team'], 'team2': inn2['team'],
        'team1_runs': inn1['runs'], 'team1_wickets': inn1['wickets'],
        'team1_extras': inn1['extras'], 'team1_fours': inn1['fours'],
        'team1_sixes': inn1['sixes'], 'team1_run_rate': inn1['run_rate'],
        'team2_extras': inn2['extras'],
        'team1_won': team1_won
    })

cricket_match_df = pd.DataFrame(records)
print(f"Cricket matches: {cricket_match_df.shape[0]}")
print("Outcome dist:", cricket_match_df['team1_won'].value_counts().to_dict())

le_team1 = LabelEncoder()
le_team2 = LabelEncoder()
cricket_match_df['team1_enc'] = le_team1.fit_transform(cricket_match_df['team1'])
cricket_match_df['team2_enc'] = le_team2.fit_transform(cricket_match_df['team2'])

cricket_features = ['team1_enc', 'team2_enc', 'team1_runs', 'team1_wickets',
                     'team1_extras', 'team1_fours', 'team1_sixes',
                     'team1_run_rate', 'team2_extras']

X_ck = cricket_match_df[cricket_features].copy()
y_ck = cricket_match_df['team1_won'].values

X_ck_train, X_ck_test, y_ck_train, y_ck_test = train_test_split(
    X_ck, y_ck, test_size=0.2, random_state=SEED, stratify=y_ck
)

scaler_ck = StandardScaler()
X_ck_train_s = scaler_ck.fit_transform(X_ck_train)
X_ck_test_s = scaler_ck.transform(X_ck_test)

print(f"Train: {X_ck_train.shape}  Test: {X_ck_test.shape}")

# --- Random Forest (Cricket) ---
rf_ck = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=SEED)
rf_ck.fit(X_ck_train, y_ck_train)
rf_ck_pred = rf_ck.predict(X_ck_test)

rf_ck_metrics = {
    "accuracy": round(accuracy_score(y_ck_test, rf_ck_pred), 4),
    "precision": round(precision_score(y_ck_test, rf_ck_pred, average='macro', zero_division=0), 4),
    "recall": round(recall_score(y_ck_test, rf_ck_pred, average='macro', zero_division=0), 4),
    "f1_score": round(f1_score(y_ck_test, rf_ck_pred, average='macro', zero_division=0), 4),
}
rf_ck_cm = confusion_matrix(y_ck_test, rf_ck_pred).tolist()
rf_ck_importance = {feat: round(imp, 4) for feat, imp in
                     zip(cricket_features, rf_ck.feature_importances_)}

print(f"RF Cricket — Acc: {rf_ck_metrics['accuracy']}")

# --- ANN (Cricket) ---
ann_ck_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
ann_ck_cm = [[0]*2]*2

if HAS_TF:
    ann_ck = keras.Sequential([
        layers.Input(shape=(X_ck_train_s.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    ann_ck.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop_ck = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True)

    ann_ck.fit(X_ck_train_s, y_ck_train, validation_split=0.15,
               epochs=150, batch_size=16, callbacks=[early_stop_ck], verbose=0)

    ann_ck_probs = ann_ck.predict(X_ck_test_s, verbose=0).ravel()
    ann_ck_pred = (ann_ck_probs >= 0.5).astype(int)

    ann_ck_metrics = {
        "accuracy": round(accuracy_score(y_ck_test, ann_ck_pred), 4),
        "precision": round(precision_score(y_ck_test, ann_ck_pred, average='macro', zero_division=0), 4),
        "recall": round(recall_score(y_ck_test, ann_ck_pred, average='macro', zero_division=0), 4),
        "f1_score": round(f1_score(y_ck_test, ann_ck_pred, average='macro', zero_division=0), 4),
    }
    ann_ck_cm = confusion_matrix(y_ck_test, ann_ck_pred).tolist()
    ann_ck.save(os.path.join(MODEL_DIR, "ann_cricket.keras"))
    print(f"ANN Cricket — Acc: {ann_ck_metrics['accuracy']}")
else:
    print("Skipping ANN Cricket (no TensorFlow)")

# ===========================================================================
# SAVE EVERYTHING
# ===========================================================================
print("\n" + "="*60)
print("SAVING MODELS AND METADATA")
print("="*60)

# Save sklearn models
joblib.dump(rf_fb, os.path.join(MODEL_DIR, "rf_football.joblib"))
joblib.dump(rf_ck, os.path.join(MODEL_DIR, "rf_cricket.joblib"))
joblib.dump(scaler_fb, os.path.join(MODEL_DIR, "scaler_football.joblib"))
joblib.dump(scaler_ck, os.path.join(MODEL_DIR, "scaler_cricket.joblib"))
joblib.dump(le_fb, os.path.join(MODEL_DIR, "le_football.joblib"))
joblib.dump(le_team1, os.path.join(MODEL_DIR, "le_cricket_team1.joblib"))
joblib.dump(le_team2, os.path.join(MODEL_DIR, "le_cricket_team2.joblib"))

# Save cricket match-level data for the backend to use
cricket_match_df.to_csv(os.path.join(MODEL_DIR, "cricket_match_data.csv"), index=False)

# Compute football team averages for prediction
home_avgs = football_df.groupby('HomeTeam')[['HS','HST','HF','HC','HY','HR']].mean()
away_avgs = football_df.groupby('AwayTeam')[['AS','AST','AF','AC','AY','AR']].mean()
odds_avg = football_df[['B365H','B365D','B365A']].mean().to_dict()

fb_team_stats = {}
for team in sorted(football_df['HomeTeam'].unique()):
    h = home_avgs.loc[team].to_dict() if team in home_avgs.index else {}
    a = away_avgs.loc[team].to_dict() if team in away_avgs.index else {}
    fb_team_stats[team] = {**h, **a}

# Compute cricket team averages
ck_batting_first_avgs = cricket_match_df.groupby('team1').agg({
    'team1_runs': 'mean', 'team1_wickets': 'mean', 'team1_extras': 'mean',
    'team1_fours': 'mean', 'team1_sixes': 'mean', 'team1_run_rate': 'mean'
}).to_dict('index')

ck_chasing_avgs = cricket_match_df.groupby('team2').agg({
    'team2_extras': 'mean'
}).to_dict('index')

# Compute EDA statistics for the frontend
# Football
fb_ftr_dist = football_df['FTR'].value_counts().to_dict()
fb_goals_home = football_df['FTHG'].describe().to_dict()
fb_goals_away = football_df['FTAG'].describe().to_dict()
fb_team_goals = football_df.groupby('HomeTeam')['FTHG'].sum().add(
    football_df.groupby('AwayTeam')['FTAG'].sum(), fill_value=0
).sort_values(ascending=False).to_dict()
fb_shots_corr = football_df[['HS','AS','HST','AST','HF','AF','HC','AC',
                              'HY','AY','HR','AR','FTHG','FTAG']].corr().round(3).to_dict()
fb_home_away_goals = {
    'home_goals_per_match': round(football_df['FTHG'].mean(), 2),
    'away_goals_per_match': round(football_df['FTAG'].mean(), 2),
    'total_goals': int(football_df['FTHG'].sum() + football_df['FTAG'].sum()),
    'total_matches': len(football_df),
    'home_wins': int(fb_ftr_dist.get('H', 0)),
    'draws': int(fb_ftr_dist.get('D', 0)),
    'away_wins': int(fb_ftr_dist.get('A', 0)),
}

# Per-team stats for football
fb_team_detailed = {}
for team in sorted(football_df['HomeTeam'].unique()):
    home_matches = football_df[football_df['HomeTeam'] == team]
    away_matches = football_df[football_df['AwayTeam'] == team]
    total_matches = len(home_matches) + len(away_matches)
    wins = len(home_matches[home_matches['FTR'] == 'H']) + len(away_matches[away_matches['FTR'] == 'A'])
    draws = len(home_matches[home_matches['FTR'] == 'D']) + len(away_matches[away_matches['FTR'] == 'D'])
    losses = total_matches - wins - draws
    goals_scored = int(home_matches['FTHG'].sum() + away_matches['FTAG'].sum())
    goals_conceded = int(home_matches['FTAG'].sum() + away_matches['FTHG'].sum())
    fb_team_detailed[team] = {
        'matches': total_matches, 'wins': wins, 'draws': draws, 'losses': losses,
        'goals_scored': goals_scored, 'goals_conceded': goals_conceded,
        'goal_diff': goals_scored - goals_conceded,
        'avg_shots': round((home_matches['HS'].mean() + away_matches['AS'].mean()) / 2, 1),
        'avg_shots_on_target': round((home_matches['HST'].mean() + away_matches['AST'].mean()) / 2, 1),
    }

# Cricket EDA
ck_total_runs = int(cricket_raw['total_runs'].sum())
ck_total_wickets = int(cricket_raw['is_wicket'].sum())
ck_total_fours = int((cricket_raw['batsman_runs'] == 4).sum())
ck_total_sixes = int((cricket_raw['batsman_runs'] == 6).sum())
ck_total_matches = cricket_match_df.shape[0]

# Top run scorers (batters)
ck_top_batters = cricket_raw.groupby('batter')['batsman_runs'].sum().sort_values(
    ascending=False).head(15).to_dict()

# Top wicket takers
ck_top_bowlers = cricket_raw[cricket_raw['is_wicket'] == 1].groupby(
    'bowler')['is_wicket'].sum().sort_values(ascending=False).head(15).to_dict()

# Runs per over distribution
ck_runs_per_over = cricket_raw.groupby('over')['total_runs'].mean().round(2).to_dict()

# Team-wise total runs
ck_team_runs = cricket_raw.groupby('batting_team')['total_runs'].sum().sort_values(
    ascending=False).to_dict()

# Wickets per over
ck_wickets_per_over = cricket_raw.groupby('over')['is_wicket'].mean().round(4).to_dict()

# Outcome distribution
ck_outcome_dist = cricket_match_df['team1_won'].value_counts().to_dict()

# Ball-by-ball correlation
ck_ball_corr = cricket_raw[['over','ball','batsman_runs','extra_runs',
                             'total_runs','is_wicket']].corr().round(3).to_dict()

# Win rates by team (as batting first)
ck_team_win_rates = {}
for team in cricket_match_df['team1'].unique():
    t = cricket_match_df[cricket_match_df['team1'] == team]
    if len(t) > 0:
        ck_team_win_rates[team] = {
            'bat_first_matches': len(t),
            'bat_first_wins': int(t['team1_won'].sum()),
            'bat_first_win_rate': round(t['team1_won'].mean() * 100, 1)
        }

# Save metadata
metadata = {
    "football": {
        "features": football_features,
        "classes": list(le_fb.classes_),
        "teams": sorted(football_df['HomeTeam'].unique().tolist()),
        "team_stats": fb_team_stats,
        "odds_avg": odds_avg,
        "rf_metrics": rf_fb_metrics,
        "ann_metrics": ann_fb_metrics,
        "rf_confusion_matrix": rf_fb_cm,
        "ann_confusion_matrix": ann_fb_cm,
        "rf_feature_importance": rf_fb_importance,
        "dataset_stats": fb_home_away_goals,
        "ftr_distribution": fb_ftr_dist,
        "team_detailed": fb_team_detailed,
        "correlation": fb_shots_corr,
    },
    "cricket": {
        "features": cricket_features,
        "classes": ["Chasing Team Won", "Batting First Won"],
        "teams_batting_first": sorted(le_team1.classes_.tolist()),
        "teams_chasing": sorted(le_team2.classes_.tolist()),
        "all_teams": sorted(set(le_team1.classes_.tolist()) | set(le_team2.classes_.tolist())),
        "batting_first_avgs": {k: {kk: round(vv, 2) for kk, vv in v.items()}
                                for k, v in ck_batting_first_avgs.items()},
        "chasing_avgs": {k: {kk: round(vv, 2) for kk, vv in v.items()}
                          for k, v in ck_chasing_avgs.items()},
        "rf_metrics": rf_ck_metrics,
        "ann_metrics": ann_ck_metrics,
        "rf_confusion_matrix": rf_ck_cm,
        "ann_confusion_matrix": ann_ck_cm,
        "rf_feature_importance": rf_ck_importance,
        "dataset_stats": {
            "total_deliveries": int(cricket_raw.shape[0]),
            "total_matches": ck_total_matches,
            "total_runs": ck_total_runs,
            "total_wickets": ck_total_wickets,
            "total_fours": ck_total_fours,
            "total_sixes": ck_total_sixes,
            "teams_count": len(set(le_team1.classes_.tolist()) | set(le_team2.classes_.tolist())),
        },
        "outcome_distribution": {str(k): v for k, v in ck_outcome_dist.items()},
        "top_batters": ck_top_batters,
        "top_bowlers": ck_top_bowlers,
        "runs_per_over": {str(k): v for k, v in ck_runs_per_over.items()},
        "wickets_per_over": {str(k): v for k, v in ck_wickets_per_over.items()},
        "team_runs": ck_team_runs,
        "team_win_rates": ck_team_win_rates,
        "ball_correlation": ck_ball_corr,
    },
    "has_tensorflow": HAS_TF,
}

with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2, default=str)

print("\nAll models and metadata saved to:", MODEL_DIR)
print("Files created:")
for f_name in os.listdir(MODEL_DIR):
    f_path = os.path.join(MODEL_DIR, f_name)
    size = os.path.getsize(f_path)
    print(f"  {f_name:40s} {size:>10,} bytes")

print("\n[SUCCESS] Training complete!")
