"""
TESS Exoplanet Classification Model Implementation
Uses the best ensemble model trained from AdvancedEnsembleExoplanetClassifier
"""

import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the best ensemble model package
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "voting_hard.pkl")

try:
    model_package = joblib.load(model_path)
    print("✓ Model package loaded successfully!")
    print(f"  Model: {model_package['model_name']}")
    print(f"  Rank: {model_package['rank']}")
    print(f"  Performance:")
    print(f"    - Accuracy: {model_package['performance']['accuracy']*100:.2f}%")
    print(f"    - F1 Score: {model_package['performance']['f1']*100:.2f}%")
    print(f"    - AUC Score: {model_package['performance']['auc']*100:.2f}%")
except FileNotFoundError:
    raise FileNotFoundError(f"⚠️  Warning: Model file not found at {model_path}")

# Extract components from model package
model = model_package['model']
scaler = model_package['scaler']
imputer = model_package['imputer']
label_encoder = model_package['label_encoder']

# Feature columns used during training
FEATURE_COLUMNS = [    
    # Stellar Proper Motion
    'st_pmra',  # PMRA [mas/yr]
    'st_pmraerr1',  # PMRA Upper Unc [mas/yr]
    'st_pmraerr2',  # PMRA Lower Unc [mas/yr]
    'st_pmdec',  # PMDec [mas/yr]
    'st_pmdecerr1',  # PMDec Upper Unc [mas/yr]
    'st_pmdecerr2',  # PMDec Lower Unc [mas/yr]
    
    # Orbital and Transit Parameters
    'pl_tranmid',  # Planet Transit Midpoint Value [BJD]
    'pl_orbper',  # Planet Orbital Period Value [days]
    'pl_trandurh',  # Planet Transit Duration Value [hours]
    'pl_trandurherr1',  # Planet Transit Duration Upper Unc [hours]
    'pl_trandurherr2',  # Planet Transit Duration Lower Unc [hours]
    'pl_trandep',  # Planet Transit Depth Value [ppm]
    'pl_trandeperr1',  # Planet Transit Depth Upper Unc [ppm]
    'pl_trandeperr2',  # Planet Transit Depth Lower Unc [ppm]
    
    # Planetary Properties
    'pl_rade',  # Planet Radius Value [R_Earth]
    'pl_insol',  # Planet Insolation Value [Earth flux]
    'pl_eqt',  # Planet Equilibrium Temperature Value [K]
    
    # Stellar Properties
    'st_tmag',  # TESS Magnitude
    'st_tmagerr1',  # TESS Magnitude Upper Unc
    'st_tmagerr2',  # TESS Magnitude Lower Unc
    'st_dist',  # Stellar Distance [pc]
    'st_disterr1',  # Stellar Distance Upper Unc [pc]
    'st_disterr2',  # Stellar Distance Lower Unc [pc]
    'st_teff',  # Stellar Effective Temperature Value [K]
    'st_tefferr1',  # Stellar Effective Temperature Upper Unc [K]
    'st_tefferr2',  # Stellar Effective Temperature Lower Unc [K]
    'st_logg',  # Stellar log(g) Value [cm/s**2]
    'st_rad',  # Stellar Radius Value [R_Sun]
]


def predict(**kwargs):
    """
    Predict exoplanet disposition using the best ensemble model.
    
    Parameters:
    -----------
    **kwargs : dict
        Feature values as keyword arguments. Should include all features from FEATURE_COLUMNS.
        Missing features will be handled by the imputer.
    
    Returns:
    --------
    tuple: (prediction, confidence, class_probabilities)
        - prediction (str): Predicted class label ('CONFIRMED', 'FALSE POSITIVE', or 'CANDIDATE')
        - confidence (float): Confidence score (maximum probability)
        - class_probabilities (dict): Probability for each class
    
    Example:
    --------
    >>> result = predict(
    ...     st_pmra=10.5,
    ...     st_pmraerr1=0.5,
    ...     st_pmraerr2=-0.5,
    ...     st_pmdec=5.2,
    ...     st_pmdecerr1=0.3,
    ...     st_pmdecerr2=-0.3,
    ...     pl_tranmid=2458500.0,
    ...     pl_orbper=10.5,
    ...     pl_trandurh=2.5,
    ...     pl_trandurherr1=0.1,
    ...     pl_trandurherr2=-0.1,
    ...     pl_trandep=100,
    ...     pl_trandeperr1=5,
    ...     pl_trandeperr2=-5,
    ...     pl_rade=2.0,
    ...     pl_insol=1.0,
    ...     pl_eqt=500,
    ...     st_tmag=10.5,
    ...     st_tmagerr1=0.1,
    ...     st_tmagerr2=-0.1,
    ...     st_dist=100.0,
    ...     st_disterr1=5.0,
    ...     st_disterr2=-5.0,
    ...     st_teff=5500,
    ...     st_tefferr1=100,
    ...     st_tefferr2=-100,
    ...     st_logg=4.5,
    ...     st_rad=1.0
    ... )
    >>> print(f"Prediction: {result[0]} (Confidence: {result[1]:.2%})")
    """
    
    # Create DataFrame with proper column order
    input_data = {}
    for feature in FEATURE_COLUMNS:
        input_data[feature] = kwargs.get(feature, np.nan)
    
    X_input = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
    
    # 1. Impute missing values
    X_imputed = imputer.transform(X_input)
    
    # 2. Scale features
    X_scaled = scaler.transform(X_imputed)
    
    prediction_encoded = model.predict(X_scaled)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Get prediction probabilities
    try:
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = float(np.max(probabilities))
        
        # Create probability dictionary for all classes
        class_probabilities = {}
        for i, prob in enumerate(probabilities):
            class_label = label_encoder.inverse_transform([i])[0]
            class_probabilities[class_label] = float(prob)
        
    except AttributeError:
        # Model doesn't support probability prediction
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
    predictions_encoded = model.predict(X_scaled)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    
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
        for i, class_label in enumerate(label_encoder.classes_):
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
        'classes': list(label_encoder.classes_),
        'model_type': type(model).__name__
    }
