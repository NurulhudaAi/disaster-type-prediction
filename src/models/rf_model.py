import json
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional
from src.models.base import BaseDisasterModel
from src.models.utils import CLASS_NAMES, format_human_explanations

class RandomForestModel(BaseDisasterModel):
    def __init__(self, pipeline_path: str = "models/rf_pipeline.joblib", meta_path: str = "models/multi_model_metadata.json"):
        self.pipeline = joblib.load(pipeline_path)
        self.estimator = self.pipeline.named_steps["random_forest"]
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_names = meta["feature_names"]
        self._metrics = meta["leaderboard"]["Random Forest"]
        self.explainer = shap.TreeExplainer(self.estimator)

    @property
    def name(self) -> str:
        return "Random Forest"

    @property
    def model_type(self) -> str:
        return "Bagging Ensemble (SMOTETomek Pipeline)"

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        df = pd.DataFrame([[features.get(f, 0.0) for f in self.feature_names]], columns=self.feature_names)
        pred_id = int(self.pipeline.predict(df)[0])
        probs = self.pipeline.predict_proba(df)[0]
        return {
            "predicted_id": pred_id,
            "predicted_class": CLASS_NAMES[pred_id],
            "confidence": round(float(probs[pred_id]), 4),
            "probabilities": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}
        }

    def explain(self, features: Dict[str, float], target_class_idx: Optional[int] = None, top_k: int = 5) -> Dict[str, Any]:
        df = pd.DataFrame([[features.get(f, 0.0) for f in self.feature_names]], columns=self.feature_names)
        if target_class_idx is None:
            pred_id = int(self.pipeline.predict(df)[0])
        else:
            pred_id = target_class_idx

        raw_shap = self.explainer.shap_values(df)
        if isinstance(raw_shap, list):
            class_shap = raw_shap[pred_id][0]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            class_shap = raw_shap[0, :, pred_id]
        else:
            class_shap = raw_shap[0]

        contributions = []
        for feat, val, s_val in zip(self.feature_names, df.iloc[0], class_shap):
            contributions.append({
                "feature": feat,
                "value": float(val),
                "shap_value": round(float(s_val), 4),
                "direction": "positive" if s_val > 0 else "negative",
                "abs_impact": abs(float(s_val))
            })

        sorted_contribs = sorted(contributions, key=lambda x: x["abs_impact"], reverse=True)
        top_pos = [c for c in sorted_contribs if c["direction"] == "positive"][:top_k]
        top_neg = [c for c in sorted_contribs if c["direction"] == "negative"][:top_k]
        top_all = sorted_contribs[:top_k]
        target_name = CLASS_NAMES[pred_id]

        return {
            "target_class": target_name,
            "target_id": pred_id,
            "top_positive": top_pos,
            "top_negative": top_neg,
            "top_all": top_all,
            "human_explanations": format_human_explanations(top_all, target_name)
        }

    def metrics(self) -> Dict[str, Any]:
        return self._metrics
