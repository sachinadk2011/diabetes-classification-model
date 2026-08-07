import numpy as np
import shap
import joblib
from app.core import logger
import cloudpickle

with open('models/diabetes_rf_model.pkl', 'rb') as f:
    model = cloudpickle.load(f)
# model = joblib.load("models/diabetes_rf_model.pkl")
rf_model = model.pipeline.named_steps["model"]

explainer = shap.TreeExplainer(rf_model) # works directly with RF, XGBoost, LightGBM etc.

def into_dataframe(input_data: list, feature_names: list):
    """Convert input data into a pandas DataFrame."""
    import pandas as pd
    df = pd.DataFrame([input_data], columns=feature_names)
    logger.info(f"Input data converted to DataFrame: {df}")
    return df

def get_top_factors(feature_array: list, feature_names: list, top_n: int = 3):
    X = into_dataframe(feature_array, feature_names)
    logger.info(f"Calculating SHAP values for input data: {X}")
    shap_values = explainer.shap_values(X)
    logger.info(f"SHAP values calculated: {shap_values}")
    # For binary classification, shap_values is often a list [class_0_values, class_1_values]
    # You want the values pushing toward class 1 (diabetic)
    if isinstance(shap_values, list):
        values = shap_values[1][0]  # class 1, first (only) row
    else:
        values = shap_values[0][:, 1]

    contributions = []
    logger.info(f"SHAP values: {values}")
    for name, val, shap_val in zip(feature_names, feature_array, values):
        sv = float(shap_val)
        contributions.append({
            "feature": name,
            "value": val,
            "impact": "increased risk" if sv > 0 else "decreased risk",
            "weight": round(abs(sv), 3)
        })

    contributions.sort(key=lambda x: x["weight"], reverse=True)
    return contributions[:top_n]

def predict(input_data: list, feature_names: list = None):
    input_array = into_dataframe(input_data, feature_names)
    prediction = model.predict_risk(input_array)
    
    logger.info(f"Prediction made for input data: {input_array}")
    logger.info(f"Prediction result: {prediction[0]}")
    return int(prediction[0])  # return the first (and only) prediction


def predict_probability(input_data: list, feature_names: list = None):
    input_array = into_dataframe(input_data, feature_names)
    probability = model.predict_proba(input_array)
    logger.info(f"Probability prediction made for input data: {input_array}")
    logger.info(f"Probability result: {probability[0][1]}")  # probability of class 1 (diabetic)
    return float(probability[0][1])  # return the probability of class 1 (diabetic)

def get_overall_response(input_data: list, feature_names: list):

    prediction = predict(input_data, feature_names)
    probability = predict_probability(input_data, feature_names)
    top_factors = get_top_factors(input_data, feature_names)

    overall_response = {
        "prediction": prediction,
        "diabetes_probability": probability,
        "top_contributing_factors": top_factors if top_factors else None
        
    }

    logger.info(f"Overall response generated: {overall_response}")
    return overall_response