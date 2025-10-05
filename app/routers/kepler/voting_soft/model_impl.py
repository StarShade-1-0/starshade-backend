"""
Kepler Exoplanet Classification Model Implementation
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
model_path = os.path.join(current_dir, "voting_soft.pkl")

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
    # Orbital and Transit Parameters
    'koi_period',  # Orbital Period [days]
    'koi_time0bk',  # Transit Epoch [BKJD]
    'koi_time0',  # Transit Epoch [BJD]
    'koi_impact',  # Impact Parameter
    'koi_impact_err1',  # Impact Parameter Upper Unc.
    'koi_impact_err2',  # Impact Parameter Lower Unc.
    'koi_duration',  # Transit Duration [hrs]
    'koi_duration_err1',  # Transit Duration Upper Unc. [hrs]
    'koi_duration_err2',  # Transit Duration Lower Unc. [hrs]
    'koi_depth',  # Transit Depth [ppm]
    'koi_depth_err1',  # Transit Depth Upper Unc. [ppm]
    'koi_depth_err2',  # Transit Depth Lower Unc. [ppm]
    
    # Planet-Star Ratios
    'koi_ror',  # Planet-Star Radius Ratio
    'koi_ror_err1',  # Planet-Star Radius Ratio Upper Unc.
    'koi_ror_err2',  # Planet-Star Radius Ratio Lower Unc.
    'koi_srho',  # Fitted Stellar Density [g/cm**3]
    'koi_srho_err1',  # Fitted Stellar Density Upper Unc.
    'koi_srho_err2',  # Fitted Stellar Density Lower Unc.
    
    # Planetary Properties
    'koi_prad',  # Planetary Radius [Earth radii]
    'koi_prad_err1',  # Planetary Radius Upper Unc.
    'koi_prad_err2',  # Planetary Radius Lower Unc.
    'koi_sma',  # Orbit Semi-Major Axis [au]
    'koi_incl',  # Inclination [deg]
    'koi_teq',  # Equilibrium Temperature [K]
    'koi_insol',  # Insolation Flux [Earth flux]
    'koi_insol_err1',  # Insolation Flux Upper Unc.
    'koi_insol_err2',  # Insolation Flux Lower Unc.
    'koi_dor',  # Planet-Star Distance over Star Radius
    'koi_dor_err1',  # Planet-Star Distance over Star Radius Upper Unc.
    'koi_dor_err2',  # Planet-Star Distance over Star Radius Lower Unc.
    
    # Limb Darkening
    'koi_ldm_coeff2',  # Limb Darkening Coeff. 2
    'koi_ldm_coeff1',  # Limb Darkening Coeff. 1
    
    # Statistics
    'koi_max_sngle_ev',  # Maximum Single Event Statistic
    'koi_max_mult_ev',  # Maximum Multiple Event Statistic
    'koi_model_snr',  # Transit Signal-to-Noise
    'koi_count',  # Number of Planets
    'koi_num_transits',  # Number of Transits
    'koi_bin_oedp_sig',  # Odd-Even Depth Comparison Statistic
    
    # Stellar Properties
    'koi_steff',  # Stellar Effective Temperature [K]
    'koi_steff_err1',  # Stellar Effective Temperature Upper Unc.
    'koi_steff_err2',  # Stellar Effective Temperature Lower Unc.
    'koi_slogg',  # Stellar Surface Gravity [log10(cm/s**2)]
    'koi_slogg_err1',  # Stellar Surface Gravity Upper Unc.
    'koi_slogg_err2',  # Stellar Surface Gravity Lower Unc.
    'koi_srad',  # Stellar Radius [Solar radii]
    'koi_srad_err1',  # Stellar Radius Upper Unc.
    'koi_srad_err2',  # Stellar Radius Lower Unc.
    'koi_smass',  # Stellar Mass [Solar mass]
    'koi_smass_err1',  # Stellar Mass Upper Unc.
    'koi_smass_err2',  # Stellar Mass Lower Unc.
    'koi_fwm_stat_sig',  # FW Offset Significance [percent]
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
    ...     koi_period=10.5,
    ...     koi_time0bk=131.5,
    ...     koi_impact=0.5,
    ...     koi_duration=2.5,
    ...     koi_depth=100,
    ...     koi_prad=2.0,
    ...     koi_teq=500,
    ...     koi_insol=1.0,
    ...     koi_model_snr=20,
    ...     koi_tce_plnt_num=1,
    ...     koi_steff=5500,
    ...     koi_slogg=4.5,
    ...     koi_srad=1.0,
    ...     ra=290.0,
    ...     dec=45.0,
    ...     koi_kepmag=15.0
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
