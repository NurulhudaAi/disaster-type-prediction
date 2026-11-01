import json
import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

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

def main():
    print("Loading processed dataset...")
    data_path = "data/processed/disaster_processed_feature_selected.csv"
    data = pd.read_csv(data_path)
    
    X = data.drop(columns=["disaster_type_encoded"])
    y = data["disaster_type_encoded"]
    
    feature_names = X.columns.tolist()
    print(f"Total features: {len(feature_names)}")
    print(f"Total records: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    print("\nInitializing Best Random Forest Pipeline...")
    rf_pipeline = Pipeline([
        ("smotetomek", SMOTETomek(random_state=42)),
        ("random_forest", RandomForestClassifier(
            n_estimators=100,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features="log2",
            max_depth=40,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("Fitting model (this applies SMOTETomek resampling then trains Random Forest)...")
    rf_pipeline.fit(X_train, y_train)
    
    print("\nEvaluating on Test Set...")
    y_pred = rf_pipeline.predict(X_test)
    y_proba = rf_pipeline.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy: {acc:.4f} (Goal: ~0.9156)")
    print(f"F1-Macro: {f1_macro:.4f} (Goal: ~0.8507)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    
    # 1. Save pipeline
    model_path = "models/rf_pipeline.joblib"
    print(f"Saving model pipeline to {model_path}...")
    joblib.dump(rf_pipeline, model_path)
    
    # 2. Save metadata
    regions = [c.replace("region_", "") for c in feature_names if c.startswith("region_")]
    countries = [c.replace("country_", "") for c in feature_names if c.startswith("country_")]
    scales = [c.replace("magnitude_scale_", "") for c in feature_names if c.startswith("magnitude_scale_")]
    
    metadata = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "class_names": CLASS_NAMES,
        "class_mapping": {i: name for i, name in enumerate(CLASS_NAMES)},
        "regions": regions,
        "countries": sorted(countries),
        "magnitude_scales": scales,
        "metrics": {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro)
        }
    }
    
    with open("models/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("Saved models/metadata.json")

    # 3. Build Ground Truth Test Bank for interactive evaluation
    print("\nBuilding Ground Truth Test Bank for Agent evaluation...")
    test_indices = X_test.index.tolist()
    bank = []
    
    for i, idx in enumerate(test_indices):
        true_class_idx = int(y_test.iloc[i])
        pred_class_idx = int(y_pred[i])
        probs = [float(p) for p in y_proba[i]]
        confidence = probs[pred_class_idx]
        is_correct = (true_class_idx == pred_class_idx)
        
        # Extract meaningful human-readable attributes
        row = X_test.iloc[i]
        
        # Determine country
        c_name = "Unknown"
        for c in countries:
            if row.get(f"country_{c}", 0) == 1:
                c_name = c
                break
                
        # Determine region
        r_name = "Unknown"
        for r in regions:
            if row.get(f"region_{r}", 0) == 1:
                r_name = r
                break
                
        # Determine magnitude scale
        scale_name = "Unknown"
        for s in scales:
            if row.get(f"magnitude_scale_{s}", 0) == 1:
                scale_name = s
                break

        case = {
            "case_id": f"CASE-{i+1:04d}",
            "original_index": int(idx),
            "ground_truth_class": CLASS_NAMES[true_class_idx],
            "ground_truth_id": true_class_idx,
            "predicted_class": CLASS_NAMES[pred_class_idx],
            "predicted_id": pred_class_idx,
            "confidence": round(confidence, 4),
            "is_correct": is_correct,
            "probabilities": {CLASS_NAMES[k]: round(probs[k], 4) for k in range(len(CLASS_NAMES))},
            "summary": {
                "country": c_name,
                "region": r_name,
                "start_month": int(row.get("start_month", 0)),
                "magnitude_scale": scale_name,
                "magnitude_missing": bool(row.get("magnitude_missing", 0)),
                "magnitude_scaled": round(float(row.get("magnitude_scaled", 0.0)), 4),
                "duration_norm": round(float(row.get("event_duration_log_norm", 0.0)), 4)
            },
            "features": {k: float(row[k]) for k in feature_names}
        }
        bank.append(case)
        
    print(f"Total test cases generated: {len(bank)}")
    # Sample balanced set of cases: mix of correct and incorrect cases across all disaster types
    correct_cases = [c for c in bank if c["is_correct"]]
    incorrect_cases = [c for c in bank if not c["is_correct"]]
    print(f"Correct cases: {len(correct_cases)}, Incorrect (Hard) cases: {len(incorrect_cases)}")
    
    with open("models/ground_truth_bank.json", "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)
    print("Saved models/ground_truth_bank.json successfully!")

if __name__ == "__main__":
    main()
