import json
import os
import joblib
import numpy as np
import pandas as pd

class DisasterPredictor:
    """Production predictor wrapper for the trained Random Forest Pipeline."""
    
    def __init__(self, model_path="models/rf_pipeline.joblib", metadata_path="models/metadata.json"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run export_model.py first.")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")
            
        self.pipeline = joblib.load(model_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        self.feature_names = self.metadata["feature_names"]
        self.class_names = self.metadata["class_names"]
        self.class_mapping = {int(k): v for k, v in self.metadata["class_mapping"].items()}
        
    def predict_sample(self, feature_dict):
        """
        Takes a dictionary of feature values, aligns to the exact 27+ model features,
        and returns prediction and probabilities.
        """
        # Create single-row DataFrame aligned with feature_names
        row_data = {feat: feature_dict.get(feat, 0.0) for feat in self.feature_names}
        df = pd.DataFrame([row_data], columns=self.feature_names)
        
        pred_idx = int(self.pipeline.predict(df)[0])
        pred_label = self.class_mapping[pred_idx]
        
        probas = self.pipeline.predict_proba(df)[0]
        prob_dict = {
            self.class_names[i]: round(float(probas[i]), 4)
            for i in range(len(self.class_names))
        }
        
        # Sort probabilities descending
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
        
        return {
            "predicted_class": pred_label,
            "predicted_id": pred_idx,
            "confidence": round(float(probas[pred_idx]), 4),
            "probabilities": sorted_probs,
            "dataframe": df
        }
