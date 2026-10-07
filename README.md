# 🏆 Sports Match Outcome Prediction (ML_Project-S4)

An end-to-end Machine Learning project to predict match outcomes for **Cricket (IPL)** and **Football (English Premier League - EPL)**, featuring exploratory data analysis, trained Random Forest models, academic documentation, and an interactive full-stack web application (**OmniMatch AI**).

---

## 📁 Repository Structure

```text
ML_Project-S4/
├── eda/                                      # Exploratory Data Analysis
│   ├── 02_EDA_EPL_2018-2019.ipynb           # Football (EPL) EDA Notebook
│   ├── 02_EDA_IPL_deliveries.ipynb          # Cricket (IPL) EDA Notebook
│   ├── EDA_Report(FB).pdf                   # Football EDA Detailed Report
│   └── EDA_Report(IPL).pdf                  # Cricket EDA Detailed Report
│
├── notebooks/                                # Machine Learning Model Training
│   └── Match_Outcome_Prediction_Cricket_Football (1).ipynb # End-to-end ML training notebook
│
├── datasets/                                 # Raw and Processed Datasets
│   ├── deliveries.csv                       # IPL Ball-by-ball Deliveries Dataset
│   └── E0 2018-2019.csv                     # English Premier League 2018-2019 Match Data
│
├── docs/                                     # Project Documentation & Presentations
│   ├── Project_Abstract.pdf                 # Project Abstract (PDF)
│   ├── Project_Abstract_Cricket_Football.docx # Project Abstract (Word document)
│   ├── Match_Outcome_Prediction_Cricket_Football.pptx # Project Presentation Slides
│   └── Literature-Review.xlsx               # Academic Literature Review & Survey
│
├── match-predictor/                          # Full-Stack Match Outcome Prediction Web App
│   ├── backend/                             # FastAPI Backend & ML Model Inference
│   │   ├── data/                            # Datasets for backend models
│   │   │   ├── deliveries.csv               # IPL Ball-by-ball delivery dataset
│   │   │   └── E0_2018-2019.csv             # EPL dataset
│   │   ├── models/                          # Serialized Scikit-Learn Models & Encoders
│   │   │   ├── rf_cricket.joblib            # Trained Random Forest Cricket Classifier
│   │   │   ├── rf_football.joblib           # Trained Random Forest Football Classifier
│   │   │   ├── scaler_cricket.joblib        # StandardScaler for Cricket features
│   │   │   ├── scaler_football.joblib       # StandardScaler for Football features
│   │   │   ├── le_cricket_team1.joblib      # Team 1 Label Encoder
│   │   │   ├── le_cricket_team2.joblib      # Team 2 Label Encoder
│   │   │   ├── le_football.joblib           # Football Team Label Encoder
│   │   │   ├── cricket_match_data.csv       # Extracted match-level features
│   │   │   └── metadata.json                # Model accuracy metrics & metadata
│   │   ├── main.py                          # FastAPI REST API & Static File Server
│   │   ├── train_models.py                  # Script to retrain and export models
│   │   └── requirements.txt                 # Python dependencies
│   ├── frontend/                            # Responsive Modern Web Interface
│   │   ├── index.html                       # Application UI & Layout
│   │   ├── style.css                        # Design System & Styling
│   │   ├── app.js                           # Frontend Application Logic & API Client
│   │   └── src/                             # Component & page modular assets
│   ├── package.json                         # Node.js runner configuration
│   └── run.js                               # Automated runner script for web app
│
├── .gitignore                               # Git ignore configuration
└── README.md                                # Project overview and documentation
```

---

## ⚡ Quick Start: Running the Match Predictor Web App

The repository includes a ready-to-run web application under [`match-predictor/`](match-predictor/) that provides an intuitive dashboard to input match scenarios and view win probabilities and analytics.

### Option 1: One-Click Launch (Node.js + Python)

From the `match-predictor` directory:
```bash
cd match-predictor
npm start
```
This automatically manages port binding and spins up the FastAPI backend on `http://127.0.0.1:8000`.

### Option 2: Running via Python directly

1. Navigate to the backend directory:
   ```bash
   cd match-predictor/backend
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```
4. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.
5. Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 📊 Exploratory Data Analysis (EDA)

The [`eda/`](eda/) directory contains in-depth data exploration for both sports:
- **Cricket (IPL)**: Analyzes toss impact, venue statistics, team run-rates across powerplays, middle overs, death overs, and historical head-to-head dynamics.
- **Football (EPL)**: Examines home advantage, half-time vs. full-time score correlation, shot conversion rates, red/yellow card disciplinary impacts, and form trends across the 2018-2019 season.

---

## 🧠 Machine Learning Methodology

- **Cricket Model**: Predicts the winning team using match venue, toss decision, current score, overs remaining, wickets down, and historical head-to-head ratings.
- **Football Model**: Predicts match outcome (Home Win, Draw, Away Win) based on recent team form, home/away strengths, half-time goal dynamics, and shot-on-target statistics.
- **Algorithm**: Tuned **Random Forest Classifiers** with feature scaling (`StandardScaler`) and categorical label encoding (`LabelEncoder`).
- Training notebooks and scripts:
  - Notebook: [`notebooks/Match_Outcome_Prediction_Cricket_Football (1).ipynb`](notebooks/)
  - Training pipeline: [`match-predictor/backend/train_models.py`](match-predictor/backend/train_models.py)

---

## 📄 Documentation & Presentation

All formal academic deliverables are neatly organized under [`docs/`](docs/):
- **Abstract**: [`Project_Abstract.pdf`](docs/Project_Abstract.pdf) and [`Project_Abstract_Cricket_Football.docx`](docs/Project_Abstract_Cricket_Football.docx)
- **Literature Review**: [`Literature-Review.xlsx`](docs/Literature-Review.xlsx)
- **Presentation Deck**: [`Match_Outcome_Prediction_Cricket_Football.pptx`](docs/Match_Outcome_Prediction_Cricket_Football.pptx)