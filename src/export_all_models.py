import json
import joblib
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb
import shap

CLASS_NAMES = [
    "Drought",
    "Earthquake",
    "Epidemic",
    "Extreme Temperature",
    "Flood",
    "Landslide",
    "Storm",
    "Wildfire"
]

def clean_cols(cols):
    return [re.sub(r'[\[\]<>{}\",:]+', '_', c) for c in cols]

def bootstrap_macro_f1_ci(y_true, y_pred, n_bootstraps=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    f1_scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        score = f1_score(y_true.iloc[idx], y_pred[idx], average="macro", zero_division=0)
        f1_scores.append(score)
    ci_lower = float(np.percentile(f1_scores, 2.5))
    ci_upper = float(np.percentile(f1_scores, 97.5))
    return round(ci_lower, 4), round(ci_upper, 4)

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    # b: A correct, B wrong
    n01 = int(np.sum(correct_a & ~correct_b))
    # c: A wrong, B correct
    n10 = int(np.sum(~correct_a & correct_b))
    total = n01 + n10
    if total == 0:
        p_val = 1.0
    else:
        # Continuity-corrected chi-squared
        chi2 = (abs(n01 - n10) - 1.0) ** 2 / total
        from scipy.stats import chi2 as chi2_dist
        p_val = float(1.0 - chi2_dist.cdf(chi2, df=1))
    return {
        "n01": n01,
        "n10": n10,
        "p_value": round(p_val, 5),
        "significant": bool(p_val < 0.05)
    }

def main():
    print("Loading feature dataset...")
    data_path = "data/processed/disaster_processed_feature_selected.csv"
    data = pd.read_csv(data_path)
    X = data.drop(columns=["disaster_type_encoded"])
    y = data["disaster_type_encoded"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 1. Load or verify Random Forest
    rf_path = "models/rf_pipeline.joblib"
    print(f"Loading Random Forest pipeline from {rf_path}...")
    rf_pipe = joblib.load(rf_path)
    y_pred_rf = rf_pipe.predict(X_test)
    y_prob_rf = rf_pipe.predict_proba(X_test)

    # 2. Train LightGBM Pipeline
    lgb_path = "models/lgb_pipeline.joblib"
    lgb_feature_names = clean_cols(feature_names)
    X_train_lgb = X_train.copy()
    X_test_lgb = X_test.copy()
    X_train_lgb.columns = lgb_feature_names
    X_test_lgb.columns = lgb_feature_names

    print("Training zero-leakage LightGBM pipeline...")
    lgb_pipe = Pipeline([
        ("smotetomek", SMOTETomek(random_state=42)),
        ("lightgbm", lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=20,
            num_leaves=50,
            min_child_samples=20,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        ))
    ])
    lgb_pipe.fit(X_train_lgb, y_train)
    joblib.dump(lgb_pipe, lgb_path)
    print(f"Saved LightGBM pipeline to {lgb_path}")
    y_pred_lgb = lgb_pipe.predict(X_test_lgb)
    y_prob_lgb = lgb_pipe.predict_proba(X_test_lgb)

    # 3. Train XGBoost Pipeline
    xgb_path = "models/xgb_pipeline.joblib"
    print("Training zero-leakage XGBoost pipeline...")
    xgb_pipe = Pipeline([
        ("smotetomek", SMOTETomek(random_state=42)),
        ("xgboost", xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.2,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss"
        ))
    ])
    xgb_pipe.fit(X_train, y_train)
    joblib.dump(xgb_pipe, xgb_path)
    print(f"Saved XGBoost pipeline to {xgb_path}")
    y_pred_xgb = xgb_pipe.predict(X_test)
    y_prob_xgb = xgb_pipe.predict_proba(X_test)

    # Metrics evaluation
    models_dict = {
        "Random Forest": (y_pred_rf, y_prob_rf, rf_pipe.named_steps["random_forest"], X_test),
        "LightGBM": (y_pred_lgb, y_prob_lgb, lgb_pipe.named_steps["lightgbm"], X_test_lgb),
        "XGBoost": (y_pred_xgb, y_prob_xgb, xgb_pipe.named_steps["xgboost"], X_test)
    }

    leaderboard = {}
    for name, (pred, prob, estimator, X_in) in models_dict.items():
        acc = float(accuracy_score(y_test, pred))
        f1_macro = float(f1_score(y_test, pred, average="macro"))
        f1_weighted = float(f1_score(y_test, pred, average="weighted"))
        ci_lower, ci_upper = bootstrap_macro_f1_ci(y_test, pred)
        report = classification_report(y_test, pred, target_names=CLASS_NAMES, output_dict=True)
        per_class_f1 = {cls: round(float(report[cls]["f1-score"]), 4) for cls in CLASS_NAMES}
        per_class_recall = {cls: round(float(report[cls]["recall"]), 4) for cls in CLASS_NAMES}
        leaderboard[name] = {
            "macro_f1": round(f1_macro, 4),
            "macro_f1_ci": [ci_lower, ci_upper],
            "accuracy": round(acc, 4),
            "weighted_f1": round(f1_weighted, 4),
            "per_class_f1": per_class_f1,
            "per_class_recall": per_class_recall
        }
        print(f"[{name}] Macro-F1: {f1_macro:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]), Acc: {acc:.4f}")

    # Significance checks (McNemar)
    mcnemar_results = {
        "Random Forest vs LightGBM": mcnemar_test(y_test, y_pred_rf, y_pred_lgb),
        "Random Forest vs XGBoost": mcnemar_test(y_test, y_pred_rf, y_pred_xgb),
        "LightGBM vs XGBoost": mcnemar_test(y_test, y_pred_lgb, y_pred_xgb),
    }

    # Precomputing SHAP explainers
    print("\nComputing SHAP explainers for test bank samples...")
    # Use TreeExplainer for each model
    rf_model = rf_pipe.named_steps["random_forest"]
    rf_explainer = shap.TreeExplainer(rf_model)
    lgb_model = lgb_pipe.named_steps["lightgbm"]
    lgb_explainer = shap.TreeExplainer(lgb_model)
    xgb_model = xgb_pipe.named_steps["xgboost"]
    xgb_explainer = shap.TreeExplainer(xgb_model)

    # Build comprehensive precomputed test bank
    regions = [c.replace("region_", "") for c in feature_names if c.startswith("region_")]
    countries = [c.replace("country_", "") for c in feature_names if c.startswith("country_")]
    scales = [c.replace("magnitude_scale_", "") for c in feature_names if c.startswith("magnitude_scale_")]

    test_indices = X_test.index.tolist()
    precomputed_bank = []
    disagreement_count = 0

    print("Building precomputed multi-model test bank (3,495 cases)...")
    for i in range(len(test_indices)):
        idx = test_indices[i]
        true_id = int(y_test.iloc[i])
        true_name = CLASS_NAMES[true_id]
        row = X_test.iloc[i]

        c_name = "Unknown"
        for c in countries:
            if row.get(f"country_{c}", 0) == 1:
                c_name = c
                break
        r_name = "Unknown"
        for r in regions:
            if row.get(f"region_{r}", 0) == 1:
                r_name = r
                break
        s_name = "Unknown"
        for s in scales:
            if row.get(f"magnitude_scale_{s}", 0) == 1:
                s_name = s
                break

        summary = {
            "country": c_name,
            "region": r_name,
            "start_month": int(row.get("start_month", 0)),
            "magnitude_scale": s_name,
            "magnitude_missing": bool(row.get("magnitude_missing", 0)),
            "magnitude_scaled": round(float(row.get("magnitude_scaled", 0.0)), 4),
            "duration_norm": round(float(row.get("event_duration_log_norm", 0.0)), 4)
        }

        # Model outputs
        m_rf_id = int(y_pred_rf[i])
        m_lgb_id = int(y_pred_lgb[i])
        m_xgb_id = int(y_pred_xgb[i])

        is_disagreement = not (m_rf_id == m_lgb_id == m_xgb_id)
        if is_disagreement:
            disagreement_count += 1

        evaluations = {
            "Random Forest": {
                "predicted_id": m_rf_id,
                "predicted_class": CLASS_NAMES[m_rf_id],
                "confidence": round(float(y_prob_rf[i][m_rf_id]), 4),
                "is_correct": bool(m_rf_id == true_id),
                "probabilities": {CLASS_NAMES[k]: round(float(y_prob_rf[i][k]), 4) for k in range(8)}
            },
            "LightGBM": {
                "predicted_id": m_lgb_id,
                "predicted_class": CLASS_NAMES[m_lgb_id],
                "confidence": round(float(y_prob_lgb[i][m_lgb_id]), 4),
                "is_correct": bool(m_lgb_id == true_id),
                "probabilities": {CLASS_NAMES[k]: round(float(y_prob_lgb[i][k]), 4) for k in range(8)}
            },
            "XGBoost": {
                "predicted_id": m_xgb_id,
                "predicted_class": CLASS_NAMES[m_xgb_id],
                "confidence": round(float(y_prob_xgb[i][m_xgb_id]), 4),
                "is_correct": bool(m_xgb_id == true_id),
                "probabilities": {CLASS_NAMES[k]: round(float(y_prob_xgb[i][k]), 4) for k in range(8)}
            }
        }

        case = {
            "case_id": f"CASE-{i+1:04d}",
            "original_index": int(idx),
            "ground_truth_id": true_id,
            "ground_truth_class": true_name,
            "summary": summary,
            "features": {k: float(row[k]) for k in feature_names},
            "evaluations": evaluations,
            "is_disagreement": is_disagreement
        }
        precomputed_bank.append(case)

    print(f"Total test cases processed: {len(precomputed_bank)}")
    print(f"Cases with model disagreement: {disagreement_count} ({disagreement_count/len(precomputed_bank)*100:.1f}%)")

    # Save to disk
    bank_path = "models/multi_model_bank.joblib"
    joblib.dump(precomputed_bank, bank_path, compress=3)
    print(f"Saved precomputed bank to {bank_path}")

    # Save metadata & leaderboard
    multi_meta = {
        "class_names": CLASS_NAMES,
        "feature_names": feature_names,
        "leaderboard": leaderboard,
        "mcnemar": mcnemar_results,
        "total_test_cases": len(precomputed_bank),
        "disagreement_cases": disagreement_count
    }
    with open("models/multi_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(multi_meta, f, indent=2, ensure_ascii=False)
    print("Saved models/multi_model_metadata.json successfully!")

if __name__ == "__main__":
    main()
