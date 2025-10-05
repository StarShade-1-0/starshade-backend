import os
import pandas as pd
import numpy as np
import joblib

# Load the model package saved by the advanced ensemble pipeline
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "stacking_logreg.pkl")
model_package = joblib.load(model_path)
print("Model package loaded:", model_package)

model = model_package['model']
scaler = model_package['scaler']
imputer = model_package['imputer']

FEATURE_COLUMNS = [
    'orbital_period',
    'transit_duration',
    'transit_duration_err1',
    'transit_duration_err2',
    'transit_depth',
    'transit_depth_err1',
    'transit_depth_err2',
    'planet_radius',
    'planet_radius_err1',
    'planet_radius_err2',
    'equi_temp',
    'stellar_temp',
    'stellar_temp_err1',
    'stellar_temp_err2',
    'stellar_radius',
    'stellar_radius_err1',
    'stellar_radius_err2'
]

def predict(**kwargs):
    """
    Predict exoplanet class using the saved advanced ensemble model.
    Input features are passed as named arguments matching FEATURE_COLUMNS.
    
    Returns predicted class label, confidence score, and class probabilities.
    """
    
    # Convert input to DataFrame
    X_input = pd.DataFrame([kwargs], columns=FEATURE_COLUMNS)
    
    # Impute missing values
    X_imputed = imputer.transform(X_input)
    
    # Scale features
    X_scaled = scaler.transform(X_imputed)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    # Get probability if available
    try:
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = float(np.max(probabilities))
        class_probabilities = {
            str(cls): float(prob)
            for cls, prob in zip(model.classes_, probabilities)
        }
    except AttributeError:
        confidence = None
        class_probabilities = None
    
    return prediction, confidence, class_probabilities
