import json
import shap
import numpy as np
import pandas as pd

class DisasterExplainer:
    """SHAP Explainer for Random Forest multi-class disaster classification."""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.pipeline = predictor.pipeline
        self.rf_model = self.pipeline.named_steps["random_forest"]
        self.feature_names = predictor.feature_names
        self.class_names = predictor.class_names
        
        # Initialize TreeExplainer
        print("Initializing SHAP TreeExplainer for Random Forest...")
        self.explainer = shap.TreeExplainer(self.rf_model)
        print("SHAP TreeExplainer ready.")
        
    def explain_sample(self, df_sample, target_class_idx=None, top_k=5):
        """
        Calculate SHAP values for a single sample DataFrame.
        Returns top features influencing the prediction.
        """
        if not isinstance(df_sample, pd.DataFrame):
            df_sample = pd.DataFrame([df_sample], columns=self.feature_names)
            
        # Ensure correct column ordering
        df_sample = df_sample[self.feature_names]
        
        # Calculate SHAP values
        raw_shap = self.explainer.shap_values(df_sample)
        
        # Determine class index to explain
        if target_class_idx is None:
            pred_idx = int(self.rf_model.predict(df_sample)[0])
        else:
            pred_idx = target_class_idx
            
        target_class_name = self.class_names[pred_idx]
        
        # In shap 0.40+, raw_shap for TreeExplainer on multiclass can be a list of arrays or 3D ndarray
        if isinstance(raw_shap, list):
            # List of arrays per class, each shape (1, n_features)
            class_shap = raw_shap[pred_idx][0]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            # Shape: (1, n_features, n_classes)
            class_shap = raw_shap[0, :, pred_idx]
        else:
            class_shap = raw_shap[0]
            
        # Combine with feature names and sample values
        contributions = []
        for feat_name, val, shap_val in zip(self.feature_names, df_sample.iloc[0], class_shap):
            # Only consider active or non-zero features or high impact features
            contributions.append({
                "feature": feat_name,
                "value": float(val),
                "shap_value": round(float(shap_val), 4),
                "impact": "positive" if shap_val > 0 else "negative",
                "abs_impact": abs(float(shap_val))
            })
            
        # Sort by absolute SHAP impact
        sorted_contributions = sorted(contributions, key=lambda x: x["abs_impact"], reverse=True)
        top_positive = [c for c in sorted_contributions if c["impact"] == "positive"][:top_k]
        top_negative = [c for c in sorted_contributions if c["impact"] == "negative"][:top_k]
        
        # Generate domain-friendly explanation lines in English
        explanation_bullet_points = []
        for item in top_positive:
            feat = item["feature"]
            val = item["value"]
            impact = item["shap_value"]
            if feat.startswith("magnitude_scale_") and val == 1:
                scale_name = feat.replace("magnitude_scale_", "")
                explanation_bullet_points.append(f"Magnitude scale unit ({scale_name}) strongly indicates {target_class_name} (SHAP +{impact:.4f})")
            elif feat.startswith("region_") and val == 1:
                reg_name = feat.replace("region_", "")
                explanation_bullet_points.append(f"Geographic region ({reg_name}) aligns with typical {target_class_name} hotspots (SHAP +{impact:.4f})")
            elif feat == "start_month":
                explanation_bullet_points.append(f"Event occurred in month {int(val)}, matching the seasonal pattern of {target_class_name} (SHAP +{impact:.4f})")
            elif "duration" in feat:
                explanation_bullet_points.append(f"Recorded event duration strongly aligns with {target_class_name} dynamics (SHAP +{impact:.4f})")
            elif feat.startswith("country_") and val == 1:
                country_name = feat.replace("country_", "")
                explanation_bullet_points.append(f"Historical disaster frequency in {country_name} supports this classification (SHAP +{impact:.4f})")
            else:
                explanation_bullet_points.append(f"Feature '{feat}' (value={val}) actively drives the model towards {target_class_name} (SHAP +{impact:.4f})")
                
        return {
            "target_class": target_class_name,
            "target_id": pred_idx,
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
            "top_all_features": sorted_contributions[:top_k],
            "human_explanations": explanation_bullet_points,
            "raw_shap_vector": [float(x) for x in class_shap]
        }
