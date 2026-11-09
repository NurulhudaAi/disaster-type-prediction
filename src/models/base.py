from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseDisasterModel(ABC):
    """Common interface for all Disaster Auditor models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Short category (e.g. 'Tree Ensemble', 'Gradient Boosting', 'LLM Zero-Shot')."""
        pass

    @abstractmethod
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Run inference on a single feature record dictionary.
        Returns:
            {
                "predicted_id": int,
                "predicted_class": str,
                "confidence": float,
                "probabilities": Dict[str, float]
            }
        """
        pass

    @abstractmethod
    def explain(self, features: Dict[str, float], target_class_idx: Optional[int] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Calculate or retrieve SHAP attributions for the record.
        Returns:
            {
                "top_positive": List[Dict],
                "top_negative": List[Dict],
                "top_all": List[Dict],
                "human_explanations": List[str]
            }
        """
        pass

    @abstractmethod
    def metrics(self) -> Dict[str, Any]:
        """
        Return headline test set metrics.
        Returns:
            {
                "macro_f1": float,
                "macro_f1_ci": [float, float],
                "accuracy": float,
                "weighted_f1": float,
                "per_class_f1": Dict[str, float],
                "per_class_recall": Dict[str, float]
            }
        """
        pass
