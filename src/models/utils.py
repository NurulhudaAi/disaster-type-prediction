from typing import Dict, List, Any

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

def format_human_explanations(top_features: List[Dict[str, Any]], target_class: str) -> List[str]:
    """Translate raw feature attributions into natural language domain sentences."""
    sentences = []
    for item in top_features:
        feat = item["feature"]
        val = item["value"]
        impact = item["shap_value"]
        direction = "pushed toward" if impact > 0 else "reduced likelihood of"

        if feat == "magnitude_missing":
            if val == 1:
                sentences.append(f"Missing magnitude metric {direction} {target_class} (common for non-instrumented events, SHAP {impact:+.4f})")
            else:
                sentences.append(f"Recorded physical magnitude {direction} {target_class} (SHAP {impact:+.4f})")
        elif feat.startswith("magnitude_scale_") and val == 1:
            scale = feat.replace("magnitude_scale_", "")
            sentences.append(f"Measurement scale '{scale}' {direction} {target_class} (SHAP {impact:+.4f})")
        elif feat == "start_month":
            sentences.append(f"Event timing in Month {int(val)} {direction} {target_class} seasonality (SHAP {impact:+.4f})")
        elif feat.startswith("region_") and val == 1:
            reg = feat.replace("region_", "")
            sentences.append(f"Geographic region ({reg}) {direction} {target_class} hotspot pattern (SHAP {impact:+.4f})")
        elif feat.startswith("country_") and val == 1:
            country = feat.replace("country_", "")
            sentences.append(f"Historical disaster frequency in {country} {direction} {target_class} (SHAP {impact:+.4f})")
        elif "duration" in feat:
            sentences.append(f"Event duration log-norm {direction} {target_class} typical lifecycle (SHAP {impact:+.4f})")
        elif "magnitude_scaled" in feat:
            sentences.append(f"Standardized physical magnitude {direction} {target_class} (SHAP {impact:+.4f})")
        else:
            sentences.append(f"Feature '{feat}'={val} {direction} {target_class} (SHAP {impact:+.4f})")

    return sentences
