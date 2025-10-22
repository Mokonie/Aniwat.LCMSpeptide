#!/usr/bin/env python3.11
"""
Train Machine Learning models for peptide taste prediction
Models: Umami, Bitter
Features: Amino acid composition, Dipeptide composition, Physicochemical properties
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Amino acid properties
AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'polarity': 1, 'aromaticity': 0},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'polarity': 1, 'aromaticity': 0},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'polarity': 0, 'aromaticity': 1},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'H': {'hydrophobicity': -3.2, 'charge': 0.5, 'polarity': 1, 'aromaticity': 1},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'polarity': 1, 'aromaticity': 0},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'polarity': 1, 'aromaticity': 0},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'polarity': 1, 'aromaticity': 0},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'polarity': 1, 'aromaticity': 0},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'polarity': 1, 'aromaticity': 0},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'polarity': 1, 'aromaticity': 0},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'polarity': 0, 'aromaticity': 1},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'polarity': 1, 'aromaticity': 1},
}

def amino_acid_composition(sequence):
    """Calculate amino acid composition"""
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    composition = {}
    for aa in aa_list:
        composition[f'AAC_{aa}'] = sequence.count(aa) / len(sequence) if len(sequence) > 0 else 0
    return composition

def dipeptide_composition(sequence):
    """Calculate dipeptide composition (simplified - only common dipeptides)"""
    if len(sequence) < 2:
        return {}
    
    # Only use most common dipeptides to reduce feature space
    common_dipeptides = [
        'AA', 'AE', 'AD', 'AK', 'AL', 'DD', 'DE', 'DK', 'EE', 'ED', 'EK', 'EL',
        'KK', 'KE', 'KD', 'LL', 'LE', 'LD', 'GG', 'GE', 'GD', 'GL', 'GK'
    ]
    
    composition = {}
    for dp in common_dipeptides:
        composition[f'DPC_{dp}'] = sequence.count(dp) / (len(sequence) - 1) if len(sequence) > 1 else 0
    return composition

def physicochemical_properties(sequence):
    """Calculate physicochemical properties"""
    if len(sequence) == 0:
        return {
            'avg_hydrophobicity': 0,
            'avg_charge': 0,
            'avg_polarity': 0,
            'avg_aromaticity': 0,
            'length': 0
        }
    
    hydrophobicity = []
    charge = []
    polarity = []
    aromaticity = []
    
    for aa in sequence:
        if aa in AA_PROPERTIES:
            props = AA_PROPERTIES[aa]
            hydrophobicity.append(props['hydrophobicity'])
            charge.append(props['charge'])
            polarity.append(props['polarity'])
            aromaticity.append(props['aromaticity'])
    
    return {
        'avg_hydrophobicity': np.mean(hydrophobicity) if hydrophobicity else 0,
        'avg_charge': np.mean(charge) if charge else 0,
        'avg_polarity': np.mean(polarity) if polarity else 0,
        'avg_aromaticity': np.mean(aromaticity) if aromaticity else 0,
        'length': len(sequence)
    }

def extract_features(sequence):
    """Extract all features from a peptide sequence"""
    features = {}
    features.update(amino_acid_composition(sequence))
    features.update(dipeptide_composition(sequence))
    features.update(physicochemical_properties(sequence))
    return features

def train_model(data_file, model_name, output_dir='models'):
    """Train a binary classification model"""
    print(f"\n{'='*70}")
    print(f"Training {model_name} Model")
    print(f"{'='*70}")
    
    # Load data
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} samples")
    print(f"  Positive: {(df['Label'] == 1).sum()}")
    print(f"  Negative: {(df['Label'] == 0).sum()}")
    
    # Extract features
    print("\nExtracting features...")
    X_features = []
    for seq in df['Sequence']:
        features = extract_features(seq)
        X_features.append(features)
    
    # Convert to DataFrame
    X = pd.DataFrame(X_features)
    y = df['Label']
    
    print(f"Feature dimensions: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Test set performance
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(f"\nTest set accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{model_name}_model.pkl')
    feature_names_path = os.path.join(output_dir, f'{model_name}_features.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(X.columns.tolist(), feature_names_path)
    
    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Features saved: {feature_names_path}")
    
    return model, X.columns.tolist()

def main():
    print("="*70)
    print("Peptide Taste Prediction - Model Training")
    print("="*70)
    
    # Train Umami model
    umami_model, umami_features = train_model(
        'data/peptides_umami.csv',
        'umami',
        'models'
    )
    
    # Train Bitter model
    bitter_model, bitter_features = train_model(
        'data/peptides_bitter.csv',
        'bitter',
        'models'
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print("\nModels saved in 'models/' directory:")
    print("  - umami_model.pkl")
    print("  - umami_features.pkl")
    print("  - bitter_model.pkl")
    print("  - bitter_features.pkl")
    
    # Test predictions
    print("\n" + "="*70)
    print("Testing Predictions")
    print("="*70)
    
    test_peptides = ['EE', 'DD', 'DE', 'AE', 'EAGIQ', 'LPEEV']
    
    for peptide in test_peptides:
        features = extract_features(peptide)
        
        # Umami prediction
        X_umami = pd.DataFrame([features])[umami_features]
        umami_prob = umami_model.predict_proba(X_umami)[0]
        umami_pred = umami_model.predict(X_umami)[0]
        
        # Bitter prediction
        X_bitter = pd.DataFrame([features])[bitter_features]
        bitter_prob = bitter_model.predict_proba(X_bitter)[0]
        bitter_pred = bitter_model.predict(X_bitter)[0]
        
        print(f"\nPeptide: {peptide}")
        print(f"  Umami: {'Yes' if umami_pred == 1 else 'No'} (confidence: {umami_prob[1]:.2%})")
        print(f"  Bitter: {'Yes' if bitter_pred == 1 else 'No'} (confidence: {bitter_prob[1]:.2%})")

if __name__ == '__main__':
    main()

