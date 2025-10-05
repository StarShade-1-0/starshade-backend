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


def batch_predict(input_df):
    """
    Make predictions for multiple samples at once.
    
    Parameters:
    -----------
    input_df : pd.DataFrame
        DataFrame containing features for multiple samples.
        Should have columns matching FEATURE_COLUMNS.
    
    Returns:
    --------
    pd.DataFrame: DataFrame with predictions and probabilities
    """
    
    # Ensure proper column order
    X_input = input_df[FEATURE_COLUMNS].copy()
    
    # Apply preprocessing
    X_imputed = imputer.transform(X_input)
    X_scaled = scaler.transform(X_imputed)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Get probabilities if available
    try:
        probabilities = model.predict_proba(X_scaled)
        confidence_scores = np.max(probabilities, axis=1)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'prediction': predictions,
            'confidence': confidence_scores
        })
        
        # Add individual class probabilities
        for i, class_label in enumerate(model.classes_):
            result_df[f'prob_{class_label}'] = probabilities[:, i]
            
    except AttributeError:
        result_df = pd.DataFrame({
            'prediction': predictions
        })
    
    return result_df


def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
    --------
    dict: Model information including name, performance metrics, and features
    """
    
    return {
        'model_name': model_package['model_name'],
        'rank': model_package['rank'],
        'performance': model_package['performance'],
        'features': FEATURE_COLUMNS,
        'num_features': len(FEATURE_COLUMNS),
        'classes': list(model.classes_),
        'model_type': type(model).__name__
    }
