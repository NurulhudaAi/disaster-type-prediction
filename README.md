# Disaster Type Prediction & Interactive Model Auditor (Explainable AI)

A Senior Project that integrates **Multi-class Disaster Event Classification (Random Forest Pipeline)** with an **Agentic Model Auditor** and **Explainable AI (SHAP)**.

---

## 🌟 Key Features

1. **High Performance ML Pipeline**:
   - Random Forest Classifier with **SMOTETomek** class balancing and hyperparameter tuning.
   - **91.70% Test Accuracy** and **0.8518 F1-Macro** across 8 disaster categories.
2. **Interactive Ground Truth Model Auditor**:
   - Challenges the pre-trained model with 3,495 real historical events from the EM-DAT test set.
   - Live side-by-side comparison: **Model Prediction vs. Ground Truth**.
   - Real-time Session Scoreboard tracking accuracy and error patterns.
3. **Explainable AI (SHAP TreeExplainer)**:
   - Quantifies the impact of physical features (magnitude scale, event duration, timing, geographic region).
   - Explains *why* the model made each decision, both for correct cases and hard misclassified cases.
4. **Agentic Claude Integration**:
   - Uses **Anthropic Claude Tool Calling** for deep reasoning, parameter extraction, and domain insight.
   - Includes a robust **Local Auditor Engine** for offline/instant evaluation when no API key is provided.

---

## 📂 Project Structure

```text
disaster-type-prediction-1/
├── app.py                      # Streamlit Interactive Web Application
├── Dockerfile                  # Container definition for reproducible deployment
├── requirements.txt            # Python dependencies
├── models/                     # Production artifacts
│   ├── rf_pipeline.joblib      # Trained best Random Forest Pipeline
│   ├── metadata.json           # 245 feature names, class mapping, and metrics
│   └── ground_truth_bank.json  # 3,495 historical test events with ground truth
├── src/                        # Modular production code
│   ├── predictor.py            # Prediction & probability computation
│   ├── explainer.py            # SHAP TreeExplainer & feature attribution
│   ├── benchmark.py            # Ground truth bank loader & filters
│   ├── agent.py                # Claude Tool Calling & Local Auditor Agent
│   └── export_model.py         # Automated pipeline training & export script
├── notebooks/                  # Academic research & experimental record
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_EDA.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_model_export_and_benchmark.ipynb
└── data/                       # Datasets
    ├── cleaned/
    └── processed/
```

---

## 🚀 Quickstart & Usage

### 1. Run Locally (Streamlit)

Activate your Python virtual environment and run:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run with Docker

Build and launch the container with a single command:
```bash
docker build -t disaster-model-auditor .
docker run -p 8501:8501 disaster-model-auditor
```

---

## 🎯 Target Disaster Classes (8 Classes)
1. Drought
2. Earthquake
3. Epidemic
4. Extreme Temperature
5. Flood
6. Landslide
7. Storm
8. Wildfire
