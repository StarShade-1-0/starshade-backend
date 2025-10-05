import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
import joblib
warnings.filterwarnings('ignore')

# Load the model package saved by the classifier pipeline
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "stacking_rf.pkl")
model_package = joblib.load(model_path)
print("Model package loaded:", model_package)

model = model_package['model']
scaler = model_package['scaler']
imputer = model_package['imputer']
label_encoder = model_package['label_encoder']

FEATURE_COLUMNS = [
    "pl_orbper",
    "pl_tranmid",
    "pl_trandur",
    "pl_rade",
    "pl_radj",
    "pl_radjerr1",
    "pl_radjerr2",
    "pl_ratror",
    "st_rad",
    "st_raderr1",
    "st_raderr2",
    "sy_dist",
    "sy_disterr1",
    "sy_disterr2",
    "sy_plx",
    "sy_plxerr1",
    "sy_plxerr2"
]

def predict(**kwargs):
    """
    Predict exoplanet class using the stacking_rf model.
    Input features as named arguments corresponding to feature columns.
    
    Returns predicted class label and probability.
    """
    print("Received input features:\n", kwargs)

    # Convert input to DataFrame
    X_input = pd.DataFrame([kwargs], columns=FEATURE_COLUMNS)
    print("Input features:\n", X_input)
    
    # Impute missing values
    X_imputed = imputer.transform(X_input)
    print("Imputed features:\n", X_imputed)
    
    # Scale features
    X_scaled = scaler.transform(X_imputed)
    print("Scaled features:\n", X_scaled)
    

    # Make prediction
    prediction_encoded = model.predict(X_scaled)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]

    # Get probability if available
    try:
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = float(np.max(probabilities))
        class_probabilities = {
            label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(probabilities)
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