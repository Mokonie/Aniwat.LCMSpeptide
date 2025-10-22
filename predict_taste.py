#!/usr/bin/env python3.11
"""
Peptide taste prediction script
Can be called from Node.js backend
"""

import sys
import json
import pandas as pd
import numpy as np
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
    """Calculate dipeptide composition"""
    if len(sequence) < 2:
        return {}
    
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

def load_models(models_dir='models'):
    """Load trained models"""
    umami_model = joblib.load(os.path.join(models_dir, 'umami_model.pkl'))
    umami_features = joblib.load(os.path.join(models_dir, 'umami_features.pkl'))
    
    bitter_model = joblib.load(os.path.join(models_dir, 'bitter_model.pkl'))
    bitter_features = joblib.load(os.path.join(models_dir, 'bitter_features.pkl'))
    
    return {
        'umami': {'model': umami_model, 'features': umami_features},
        'bitter': {'model': bitter_model, 'features': bitter_features}
    }

def predict_taste(sequence, models):
    """Predict taste for a single peptide"""
    # Extract features
    features = extract_features(sequence)
    
    # Umami prediction
    X_umami = pd.DataFrame([features])[models['umami']['features']]
    umami_prob = models['umami']['model'].predict_proba(X_umami)[0]
    umami_pred = models['umami']['model'].predict(X_umami)[0]
    
    # Bitter prediction
    X_bitter = pd.DataFrame([features])[models['bitter']['features']]
    bitter_prob = models['bitter']['model'].predict_proba(X_bitter)[0]
    bitter_pred = models['bitter']['model'].predict(X_bitter)[0]
    
    return {
        'sequence': sequence,
        'length': len(sequence),
        'umami': {
            'predicted': bool(umami_pred == 1),
            'confidence': float(umami_prob[1])
        },
        'bitter': {
            'predicted': bool(bitter_pred == 1),
            'confidence': float(bitter_prob[1])
        },
        'sweet': {
            'predicted': False,
            'confidence': 0.0
        }
    }

def sliding_window_peptides(sequence, window_sizes=[2, 3, 4, 5], step=1):
    """Generate peptides using sliding window"""
    peptides = []
    for window_size in window_sizes:
        if len(sequence) >= window_size:
            for i in range(0, len(sequence) - window_size + 1, step):
                peptides.append(sequence[i:i+window_size])
    return list(set(peptides))  # Remove duplicates

def main():
    """Main function for CLI usage"""
    if len(sys.argv) < 2:
        print(json.dumps({
            'error': 'Usage: python predict_taste.py <sequence> [--cut]'
        }))
        sys.exit(1)
    
    sequence = sys.argv[1].upper()
    cut_mode = '--cut' in sys.argv
    
    try:
        # Load models
        models = load_models()
        
        if cut_mode:
            # Cut peptide and predict all fragments
            peptides = sliding_window_peptides(sequence)
            results = []
            
            for peptide in peptides:
                prediction = predict_taste(peptide, models)
                results.append(prediction)
            
            # Sort by highest confidence in any taste
            results.sort(key=lambda x: max(
                x['umami']['confidence'],
                x['bitter']['confidence']
            ), reverse=True)
            
            print(json.dumps({
                'original_sequence': sequence,
                'cut_mode': True,
                'total_fragments': len(results),
                'predictions': results
            }))
        else:
            # Predict single peptide
            result = predict_taste(sequence, models)
            print(json.dumps(result))
    
    except Exception as e:
        print(json.dumps({
            'error': str(e)
        }))
        sys.exit(1)

if __name__ == '__main__':
    main()

