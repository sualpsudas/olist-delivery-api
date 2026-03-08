import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_model.pkl"

# Model tek seferinde yüklenir (startup'ta)
_artifact = None


def load_model():
    global _artifact
    with open(MODEL_PATH, "rb") as f:
        _artifact = pickle.load(f)
    print(f"Model loaded. Features: {_artifact['features']}")


def get_features():
    return _artifact["features"]


def predict(data: dict) -> dict:
    features = _artifact["features"]
    model = _artifact["model"]

    df = pd.DataFrame([data])[features]
    prob_late = float(model.predict_proba(df)[0][1])
    prediction = int(prob_late >= 0.5)

    return {
        "prediction": prediction,
        "label": "Late" if prediction == 1 else "On Time",
        "probability_late": round(prob_late, 4),
        "probability_on_time": round(1 - prob_late, 4),
    }


def explain(data: dict) -> dict:
    import shap

    features = _artifact["features"]
    model = _artifact["model"]

    df = pd.DataFrame([data])[features]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    # SHAP values for class 1 (Late)
    shap_dict = {
        feat: round(float(val), 4)
        for feat, val in zip(features, shap_values[0])
    }
    # Sort by absolute importance
    shap_sorted = dict(
        sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return {
        "base_value": round(float(explainer.expected_value), 4),
        "shap_values": shap_sorted,
        "top_factor": max(shap_dict, key=lambda x: abs(shap_dict[x])),
    }
