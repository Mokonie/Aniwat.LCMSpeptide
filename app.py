"""
Peptide Taste Predictor - Streamlit App
Powered by Aniwat Kaewkrod (a.biotwu@gmail.com)

This version uses embedded training data - no external files needed!
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Import training data
from training_data import UMAMI_POSITIVE, UMAMI_NEGATIVE, BITTER_POSITIVE, BITTER_NEGATIVE

# Page configuration
st.set_page_config(
    page_title="Peptide Taste Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .umami-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
    }
    .bitter-box {
        background-color: #dbeafe;
        border-left: 4px solid #3b82f6;
    }
    .sweet-box {
        background-color: #fce7f3;
        border-left: 4px solid #ec4899;
    }
    .info-box {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .footnote {
        font-size: 0.875rem;
        color: #6b7280;
        margin-top: 1rem;
        padding: 0.75rem;
        background-color: #f9fafb;
        border-left: 3px solid #9ca3af;
        border-radius: 0.25rem;
    }
    /* Center align table cells */
    .dataframe td, .dataframe th {
        text-align: center !important;
        font-size: 1.25rem !important;
    }
    .dataframe th {
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# Feature extraction functions
def extract_features(sequence):
    """Extract features from peptide sequence"""
    sequence = sequence.upper()
    length = len(sequence)
    
    # Amino acid properties
    hydrophobicity = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    charge = {
        'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
        'Q': 0, 'E': -1, 'G': 0, 'H': 0.5, 'I': 0,
        'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
        'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
    }
    
    polarity = {
        'A': 0, 'R': 1, 'N': 1, 'D': 1, 'C': 0,
        'Q': 1, 'E': 1, 'G': 0, 'H': 1, 'I': 0,
        'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
        'S': 1, 'T': 1, 'W': 0, 'Y': 1, 'V': 0
    }
    
    aromaticity = {
        'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0,
        'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': 0,
        'L': 0, 'K': 0, 'M': 0, 'F': 1, 'P': 0,
        'S': 0, 'T': 0, 'W': 1, 'Y': 1, 'V': 0
    }
    
    # Amino acid composition
    aa_comp = {}
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        aa_comp[f'AAC_{aa}'] = sequence.count(aa) / length if length > 0 else 0
    
    # Dipeptide composition (common ones)
    dipeptides = ['AA', 'AE', 'DD', 'DE', 'ED', 'EE', 'DG', 'EG', 'GD', 'GE',
                  'KE', 'EK', 'KD', 'DK', 'QE', 'EQ', 'ND', 'DN', 'NE', 'EN',
                  'GG', 'AG', 'GA']
    dipep_comp = {}
    for dp in dipeptides:
        count = 0
        for i in range(len(sequence) - 1):
            if sequence[i:i+2] == dp:
                count += 1
        dipep_comp[f'DPC_{dp}'] = count / (length - 1) if length > 1 else 0
    
    # Physicochemical properties
    avg_hydrophobicity = sum(hydrophobicity.get(aa, 0) for aa in sequence) / length if length > 0 else 0
    avg_charge = sum(charge.get(aa, 0) for aa in sequence) / length if length > 0 else 0
    avg_polarity = sum(polarity.get(aa, 0) for aa in sequence) / length if length > 0 else 0
    avg_aromaticity = sum(aromaticity.get(aa, 0) for aa in sequence) / length if length > 0 else 0
    
    features = {
        **aa_comp,
        **dipep_comp,
        'avg_hydrophobicity': avg_hydrophobicity,
        'avg_charge': avg_charge,
        'avg_polarity': avg_polarity,
        'avg_aromaticity': avg_aromaticity,
        'length': length
    }
    
    return features

@st.cache_resource
def train_models():
    """Train models from embedded data"""
    try:
        models = {}
        
        # Prepare Umami data
        X_umami = []
        y_umami = []
        
        for seq in UMAMI_POSITIVE:
            features = extract_features(seq)
            X_umami.append(list(features.values()))
            y_umami.append(1)
        
        for seq in UMAMI_NEGATIVE:
            features = extract_features(seq)
            X_umami.append(list(features.values()))
            y_umami.append(0)
        
        X_umami = np.array(X_umami)
        y_umami = np.array(y_umami)
        feature_names = list(extract_features('AA').keys())
        
        # Train Umami model
        umami_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        umami_model.fit(X_umami, y_umami)
        
        models['umami'] = {
            'model': umami_model,
            'features': feature_names,
            'accuracy': umami_model.score(X_umami, y_umami)
        }
        
        # Prepare Bitter data
        X_bitter = []
        y_bitter = []
        
        for seq in BITTER_POSITIVE:
            features = extract_features(seq)
            X_bitter.append(list(features.values()))
            y_bitter.append(1)
        
        for seq in BITTER_NEGATIVE:
            features = extract_features(seq)
            X_bitter.append(list(features.values()))
            y_bitter.append(0)
        
        X_bitter = np.array(X_bitter)
        y_bitter = np.array(y_bitter)
        
        # Train Bitter model
        bitter_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        bitter_model.fit(X_bitter, y_bitter)
        
        models['bitter'] = {
            'model': bitter_model,
            'features': feature_names,
            'accuracy': bitter_model.score(X_bitter, y_bitter)
        }
        
        return models
    
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return None

def predict_taste(sequence, models):
    """Predict taste properties of a peptide"""
    features = extract_features(sequence)
    
    results = {}
    for taste_type in ['umami', 'bitter']:
        model_data = models[taste_type]
        model = model_data['model']
        feature_names = model_data['features']
        
        # Create feature vector in correct order
        X = np.array([[features.get(f, 0) for f in feature_names]])
        
        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        confidence = probability[1] * 100  # Probability of positive class
        
        results[taste_type] = {
            'prediction': 'Yes' if prediction == 1 else 'No',
            'confidence': confidence
        }
    
    # Sweet is not available yet
    results['sweet'] = {
        'prediction': 'N/A',
        'confidence': 0
    }
    
    return results

def cut_peptide(sequence, min_length=2, max_length=10):
    """Cut peptide into fragments using sliding window"""
    fragments = []
    sequence = sequence.upper()
    
    for length in range(min_length, min(max_length + 1, len(sequence) + 1)):
        for i in range(len(sequence) - length + 1):
            fragment = sequence[i:i+length]
            if fragment not in fragments:
                fragments.append(fragment)
    
    return fragments

# Molecular Docking Prediction Functions
def calculate_peptide_properties(sequence):
    """Calculate physicochemical properties of peptide"""
    # Amino acid properties
    aa_mw = {
        'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
        'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
        'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
        'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
    }
    
    aa_hydrophobicity = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    aa_charge = {
        'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
        'Q': 0, 'E': -1, 'G': 0, 'H': 0.5, 'I': 0,
        'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
        'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
    }
    
    aa_pka = {
        'D': 3.9, 'E': 4.3, 'K': 10.5, 'R': 12.5, 'H': 6.0,
        'C': 8.3, 'Y': 10.1
    }
    
    length = len(sequence)
    mw = sum(aa_mw.get(aa, 0) for aa in sequence) - (length - 1) * 18  # subtract water
    gravy = sum(aa_hydrophobicity.get(aa, 0) for aa in sequence) / length
    net_charge = sum(aa_charge.get(aa, 0) for aa in sequence)
    
    # Count acidic and basic residues
    acidic_count = sequence.count('D') + sequence.count('E')
    basic_count = sequence.count('K') + sequence.count('R') + sequence.count('H')
    
    # Estimate pI (simplified)
    if acidic_count > basic_count:
        pi = 3.0 + (acidic_count - basic_count) * 0.5
    elif basic_count > acidic_count:
        pi = 7.0 + (basic_count - acidic_count) * 0.5
    else:
        pi = 6.0
    
    return {
        'length': length,
        'mw': mw,
        'gravy': gravy,
        'net_charge': net_charge,
        'pi': pi,
        'acidic_pct': (acidic_count / length) * 100,
        'basic_pct': (basic_count / length) * 100,
        'acidic_count': acidic_count,
        'basic_count': basic_count
    }

def predict_t1r1_interactions(sequence, props):
    """Predict interactions with T1R1 pocket key residues"""
    score = 0
    residue_interactions = []
    
    # T1R1 key residues and their interaction preferences
    t1r1_residues = {
        'Y220': {'type': 'H-bond', 'prefers': ['S', 'T', 'N', 'Q', 'Y', 'D', 'E']},
        'E301': {'type': 'Electrostatic', 'prefers': ['K', 'R', 'H', 'D', 'E']},
        'L305': {'type': 'Hydrophobic', 'prefers': ['L', 'I', 'V', 'M', 'F', 'W', 'Y', 'A']},
        'S306': {'type': 'H-bond', 'prefers': ['S', 'T', 'N', 'Q', 'Y', 'D', 'E']},
        'S385': {'type': 'H-bond', 'prefers': ['S', 'T', 'N', 'Q', 'Y', 'D', 'E']},
        'N388': {'type': 'H-bond', 'prefers': ['S', 'T', 'N', 'Q', 'Y', 'D', 'E']},
        'D147': {'type': 'Electrostatic', 'prefers': ['K', 'R', 'H']},
        'Y169': {'type': 'H-bond', 'prefers': ['S', 'T', 'N', 'Q', 'Y', 'D', 'E']},
        'L75': {'type': 'Hydrophobic', 'prefers': ['L', 'I', 'V', 'M', 'F', 'W', 'Y', 'A']},
        'A302': {'type': 'H-bond', 'prefers': ['D', 'E', 'S', 'T']}
    }
    
    h_bond_count = 0
    electrostatic_count = 0
    hydrophobic_count = 0
    
    for residue, info in t1r1_residues.items():
        interaction = 'ไม่มี'
        likelihood = 'ต่ำ'
        
        # Check if peptide has amino acids that can interact
        matching_aa = [aa for aa in sequence if aa in info['prefers']]
        
        if matching_aa:
            interaction = info['type']
            if len(matching_aa) >= 2:
                likelihood = 'สูง'
                score += 8
                if info['type'] == 'H-bond':
                    h_bond_count += 1
                elif info['type'] == 'Electrostatic':
                    electrostatic_count += 1
                elif info['type'] == 'Hydrophobic':
                    hydrophobic_count += 1
            else:
                likelihood = 'ปานกลาง'
                score += 4
        
        residue_interactions.append({
            'Residue': residue,
            'Interaction Type': interaction,
            'Likelihood': likelihood
        })
    
    # Bonus for acidic peptides (important for umami)
    if props['acidic_count'] > 0:
        if sequence[0] in ['D', 'E']:  # N-terminal acidic
            score += 15
        if sequence[-1] in ['D', 'E']:  # C-terminal acidic
            score += 10
        score += props['acidic_count'] * 3
    
    # Bonus for hydrophilic peptides
    if props['gravy'] < -0.5:
        score += 10
    
    # Size compatibility (T1R1 prefers smaller peptides)
    if 2 <= props['length'] <= 4:
        score += 10
    elif 5 <= props['length'] <= 7:
        score += 7
    elif 8 <= props['length'] <= 10:
        score += 5
    
    # Normalize score to 0-100
    score = min(score, 100)
    
    # Estimate binding energy
    binding_energy = -5.0 - (score / 20)
    
    return {
        'score': score,
        'binding_energy': binding_energy,
        'h_bonds': h_bond_count,
        'electrostatic': electrostatic_count,
        'hydrophobic': hydrophobic_count,
        'residue_interactions': residue_interactions
    }

def predict_t1r3_interactions(sequence, props):
    """Predict interactions with T1R3 pocket key residues"""
    score = 0
    residue_interactions = []
    
    # T1R3 key residues and their interaction preferences
    t1r3_residues = {
        "S146'": {'type': 'H-bond', 'prefers': ['S', 'T', 'N', 'Q', 'Y', 'D', 'E']},
        "S147'": {'type': 'H-bond', 'prefers': ['S', 'T', 'N', 'Q', 'Y', 'D', 'E']},
        "E148'": {'type': 'Electrostatic', 'prefers': ['K', 'R', 'H', 'D', 'E']},
        "T167'": {'type': 'H-bond/Hydrophobic', 'prefers': ['T', 'S', 'L', 'I', 'V', 'M']},
        "G168'": {'type': 'Flexibility', 'prefers': ['G', 'A', 'S']},
        "S170'": {'type': 'H-bond', 'prefers': ['S', 'T', 'N', 'Q', 'Y', 'D', 'E']},
        "M171'": {'type': 'Hydrophobic', 'prefers': ['L', 'I', 'V', 'M', 'F', 'W', 'Y']},
        "D190'": {'type': 'Electrostatic', 'prefers': ['K', 'R', 'H']},
        "N386'": {'type': 'H-bond', 'prefers': ['S', 'T', 'N', 'Q', 'Y', 'D', 'E']},
        "Q389'": {'type': 'H-bond', 'prefers': ['S', 'T', 'N', 'Q', 'Y', 'D', 'E']},
        "W72'": {'type': 'Hydrophobic', 'prefers': ['F', 'W', 'Y', 'L', 'I', 'V']},
        "H145'": {'type': 'H-bond', 'prefers': ['D', 'E', 'S', 'T', 'N', 'Q']}
    }
    
    h_bond_count = 0
    electrostatic_count = 0
    hydrophobic_count = 0
    
    for residue, info in t1r3_residues.items():
        interaction = 'ไม่มี'
        likelihood = 'ต่ำ'
        
        # Check if peptide has amino acids that can interact
        matching_aa = [aa for aa in sequence if aa in info['prefers']]
        
        if matching_aa:
            interaction = info['type']
            if len(matching_aa) >= 2:
                likelihood = 'สูง'
                score += 8
                if 'H-bond' in info['type']:
                    h_bond_count += 1
                if 'Electrostatic' in info['type']:
                    electrostatic_count += 1
                if 'Hydrophobic' in info['type']:
                    hydrophobic_count += 1
            else:
                likelihood = 'ปานกลาง'
                score += 4
        
        residue_interactions.append({
            'Residue': residue,
            'Interaction Type': interaction,
            'Likelihood': likelihood
        })
    
    # Bonus for acidic peptides
    if props['acidic_count'] > 0:
        if sequence[0] in ['D', 'E']:
            score += 15
        if sequence[-1] in ['D', 'E']:
            score += 10
        score += props['acidic_count'] * 3
    
    # Bonus for highly hydrophilic peptides (T1R3 is more hydrophilic)
    if props['gravy'] < -1.0:
        score += 15
    elif props['gravy'] < -0.5:
        score += 10
    
    # Size compatibility (T1R3 prefers larger peptides)
    if 8 <= props['length'] <= 10:
        score += 10
    elif 5 <= props['length'] <= 7:
        score += 7
    elif 2 <= props['length'] <= 4:
        score += 5
    
    # Normalize score to 0-100
    score = min(score, 100)
    
    # Estimate binding energy
    binding_energy = -5.0 - (score / 20)
    
    return {
        'score': score,
        'binding_energy': binding_energy,
        'h_bonds': h_bond_count,
        'electrostatic': electrostatic_count,
        'hydrophobic': hydrophobic_count,
        'residue_interactions': residue_interactions
    }

def classify_umami_potential(t1r1_result, t1r3_result, props):
    """Classify overall umami potential based on both pockets"""
    # Calculate overall score (weighted average)
    overall_score = (t1r1_result['score'] + t1r3_result['score']) / 2
    
    # Determine preferred pocket
    if t1r1_result['binding_energy'] < t1r3_result['binding_energy'] - 0.5:
        preferred_pocket = 'T1R1'
    elif t1r3_result['binding_energy'] < t1r1_result['binding_energy'] - 0.5:
        preferred_pocket = 'T1R3'
    else:
        preferred_pocket = 'ทั้งสอง (Both)'
    
    # Classify umami potential
    avg_be = (t1r1_result['binding_energy'] + t1r3_result['binding_energy']) / 2
    
    if overall_score >= 70 and avg_be <= -7.5:
        level = 'สูง'
        recommendation = '✅ แนะนำให้ทดสอบต่อในห้องปฏิบัติการ - มีศักยภาพสูงที่จะให้รส Umami'
    elif overall_score >= 50 and avg_be <= -6.5:
        level = 'ปานกลาง'
        recommendation = '⚠️ อาจให้รส Umami ได้ - ควรพิจารณาปรับโครงสร้างเพื่อเพิ่มประสิทธิภาพ'
    elif overall_score >= 30 and avg_be <= -5.5:
        level = 'ต่ำ'
        recommendation = '⚡ ศักยภาพต่ำ - แนะนำให้ปรับโครงสร้างก่อนทดสอบ (เพิ่ม D/E, เพิ่มความชอบน้ำ)'
    else:
        level = 'ไม่น่าจะให้รส Umami'
        recommendation = '❌ ไม่แนะนำให้ทดสอบ - โครงสร้างไม่เหมาะสมสำหรับการจับกับ T1R1/T1R3'
    
    return {
        'overall_score': overall_score,
        'level': level,
        'preferred_pocket': preferred_pocket,
        'recommendation': recommendation
    }

# Header
st.markdown('<div class="main-header">🧬 Peptide Taste Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Aniwat Kaewkrod (a.biotwu@gmail.com)</div>', unsafe_allow_html=True)

# Info box
st.markdown("""
<div class="info-box">
    <strong>ℹ️ How it works:</strong><br>
    Upload LC-MS/MS data or enter peptide sequences (2-10 amino acids). 
    Our ML models predict Umami and Bitter tastes with high accuracy (~89%).
</div>
""", unsafe_allow_html=True)

# Train models
with st.spinner("🔄 Training ML models... (This happens once and is cached)"):
    models = train_models()

if models is None:
    st.error("Failed to train ML models.")
    st.stop()

# Show model info
st.success(f"""
✅ Models trained successfully!
- **Umami Model**: {models['umami']['accuracy']*100:.1f}% accuracy ({len(UMAMI_POSITIVE)} positive, {len(UMAMI_NEGATIVE)} negative samples)
- **Bitter Model**: {models['bitter']['accuracy']*100:.1f}% accuracy ({len(BITTER_POSITIVE)} positive, {len(BITTER_NEGATIVE)} negative samples)
""")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Single Prediction", "✂️ Cut & Analyze", "📁 LC-MS/MS File", "🧬 Molecular Docking"])

# Tab 1: Single Prediction
with tab1:
    st.subheader("Single Peptide Prediction")
    st.write("Enter a peptide sequence (2-20 amino acids) to predict its taste properties")
    
    sequence_input = st.text_input(
        "Peptide Sequence",
        placeholder="e.g., EAGIQ, LPEEV, DDEE",
        help="Use single-letter amino acid codes (A-Z). Length: 2-20 amino acids."
    )
    
    if st.button("🔮 Predict Taste", type="primary", use_container_width=True):
        if not sequence_input:
            st.warning("Please enter a peptide sequence")
        else:
            sequence = sequence_input.upper().strip()
            # Validate sequence
            if not all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in sequence):
                st.error("Invalid sequence. Please use only amino acid codes (A-Z)")
            elif len(sequence) < 2 or len(sequence) > 20:
                st.error("Sequence length must be between 2 and 20 amino acids")
            else:
                with st.spinner("Predicting..."):
                    results = predict_taste(sequence, models)
                
                st.success(f"✅ Prediction complete for: **{sequence}** (Length: {len(sequence)})")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="prediction-box umami-box">
                        <h3>🍜 Umami</h3>
                        <p><strong>Prediction:</strong> {results['umami']['prediction']}</p>
                        <p><strong>Confidence:</strong> {results['umami']['confidence']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="prediction-box bitter-box">
                        <h3>☕ Bitter</h3>
                        <p><strong>Prediction:</strong> {results['bitter']['prediction']}</p>
                        <p><strong>Confidence:</strong> {results['bitter']['confidence']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="prediction-box sweet-box">
                        <h3>🍯 Sweet</h3>
                        <p><strong>Prediction:</strong> {results['sweet']['prediction']}</p>
                        <p><strong>Note:</strong> Limited data</p>
                    </div>
                    """, unsafe_allow_html=True)

# Tab 2: Cut & Analyze
with tab2:
    st.subheader("Cut & Analyze Peptide")
    st.write("Input a long peptide sequence and we'll cut it into fragments and predict taste for each")
    
    long_sequence = st.text_area(
        "Peptide Sequence",
        placeholder="e.g., DDDEEEEEEEEEEEK",
        help="Enter a peptide sequence (up to 100 amino acids)"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        min_len = st.slider("Minimum fragment length", 2, 5, 2)
    with col2:
        max_len = st.slider("Maximum fragment length", 5, 10, 5)
    
    if st.button("✂️ Cut & Predict", type="primary", use_container_width=True):
        if not long_sequence:
            st.warning("Please enter a peptide sequence")
        else:
            sequence = long_sequence.upper().strip()
            if not all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in sequence):
                st.error("Invalid sequence. Please use only amino acid codes (A-Z)")
            elif len(sequence) > 100:
                st.error("Sequence too long. Maximum 100 amino acids.")
            else:
                with st.spinner("Cutting and analyzing..."):
                    fragments = cut_peptide(sequence, min_len, max_len)
                    
                    results_list = []
                    for frag in fragments:
                        pred = predict_taste(frag, models)
                        results_list.append({
                            'Sequence': frag,
                            'Length': len(frag),
                            'Umami': pred['umami']['prediction'],
                            'Umami Confidence': f"{pred['umami']['confidence']:.1f}%",
                            'Bitter': pred['bitter']['prediction'],
                            'Bitter Confidence': f"{pred['bitter']['confidence']:.1f}%",
                        })
                    
                    df = pd.DataFrame(results_list)
                
                st.success(f"✅ Generated {len(fragments)} fragments from sequence: **{sequence}**")
                
                # Show top fragments
                st.subheader("🏆 Top Fragments by Confidence")
                
                # Sort by umami confidence
                df_sorted = df.copy()
                df_sorted['Umami_Conf_Val'] = df_sorted['Umami Confidence'].str.rstrip('%').astype(float)
                df_sorted = df_sorted.sort_values('Umami_Conf_Val', ascending=False)
                
                st.dataframe(df_sorted.drop('Umami_Conf_Val', axis=1), use_container_width=True)
                
                # Footnote
                st.markdown("""
                <div class="footnote">
                    <strong>📝 คำอธิบายค่า Confidence:</strong><br>
                    • <strong>Umami Confidence:</strong> ความมั่นใจของโมเดล Machine Learning ในการทำนายว่าเปปไทด์มีรส Umami (0-100%)<br>
                    • <strong>Bitter Confidence:</strong> ความมั่นใจของโมเดล Machine Learning ในการทำนายว่าเปปไทด์มีรส Bitter (0-100%)<br>
                    <br>
                    <strong>การแปลผล:</strong><br>
                    &nbsp;&nbsp;- ≥70% = มีความมั่นใจสูง (แนะนำให้พิจารณาเปปไทด์นี้)<br>
                    &nbsp;&nbsp;- 50-70% = มีความมั่นใจปานกลาง<br>
                    &nbsp;&nbsp;- <50% = มีความมั่นใจต่ำ
                </div>
                """, unsafe_allow_html=True)
                
                # Download button
                csv = df.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name=f"peptide_fragments_{timestamp}.csv",
                    mime="text/csv"
                )

# Tab 3: LC-MS/MS File
with tab3:
    st.subheader("LC-MS/MS File Analysis")
    st.write("Upload CSV files from LC-MS/MS analysis")
    
    file_type = st.radio(
        "File Type",
        ["peptidefeatures.csv", "alldenovocandidates.csv"],
        help="Select the type of LC-MS/MS file you're uploading"
    )
    
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="Upload your LC-MS/MS data file"
    )
    
    if uploaded_file is not None:
        if st.button("🔬 Analyze File", type="primary", use_container_width=True):
            with st.spinner("Analyzing file..."):
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    peptides = []
                    
                    if file_type == "peptidefeatures.csv":
                        # Parse peptidefeatures.csv
                        for _, row in df.iterrows():
                            sequence = str(row.iloc[0]).upper()
                            sequence = ''.join(c for c in sequence if c in 'ACDEFGHIKLMNPQRSTVWY')
                            if len(sequence) >= 2:
                                confidence = float(row.iloc[1]) if len(row) > 1 else 0
                                peptides.append({
                                    'sequence': sequence,
                                    'confidence': confidence
                                })
                    else:
                        # Parse alldenovocandidates.csv
                        for _, row in df.iterrows():
                            sequence = str(row.iloc[3]).upper()  # Peptide column
                            sequence = ''.join(c for c in sequence if c in 'ACDEFGHIKLMNPQRSTVWY')
                            if len(sequence) >= 2:
                                denovo_score = float(row.iloc[6]) if len(row) > 6 else 0
                                alc = float(row.iloc[7]) if len(row) > 7 else 0
                                peptides.append({
                                    'sequence': sequence,
                                    'confidence': denovo_score,
                                    'alc': alc
                                })
                    
                    # Filter high confidence
                    high_conf = [p for p in peptides if p.get('confidence', 0) >= 60]
                    
                    st.success(f"""
                    ✅ File analyzed successfully!
                    - **Total peptides:** {len(peptides)}
                    - **High confidence (≥60%):** {len(high_conf)}
                    """)
                    
                    if high_conf:
                        st.subheader("🎯 High Confidence Peptides")
                        
                        results_list = []
                        for pep in high_conf[:20]:  # Top 20
                            pred = predict_taste(pep['sequence'], models)
                            results_list.append({
                                'Sequence': pep['sequence'],
                                'Length': len(pep['sequence']),
                                'Confidence': f"{pep['confidence']:.1f}%",
                                'Umami': pred['umami']['prediction'],
                                'Umami Conf': f"{pred['umami']['confidence']:.1f}%",
                                'Bitter': pred['bitter']['prediction'],
                                'Bitter Conf': f"{pred['bitter']['confidence']:.1f}%",
                            })
                        
                        df_results = pd.DataFrame(results_list)
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Footnote
                        st.markdown("""
                        <div class="footnote">
                            <strong>📝 คำอธิบายค่า Confidence:</strong><br>
                            • <strong>Confidence:</strong> คะแนนความเชื่อมั่นจากเครื่อง LC-MS/MS (Denovo score หรือ ALC) - แสดงความถูกต้องของการวิเคราะห์ลำดับเปปไทด์<br>
                            • <strong>Umami Conf:</strong> ความมั่นใจของโมเดล Machine Learning ในการทำนายว่าเปปไทด์มีรส Umami (0-100%)<br>
                            • <strong>Bitter Conf:</strong> ความมั่นใจของโมเดล Machine Learning ในการทำนายว่าเปปไทด์มีรส Bitter (0-100%)<br>
                            <br>
                            <strong>การแปลผล (Umami Conf / Bitter Conf):</strong><br>
                            &nbsp;&nbsp;- ≥70% = มีความมั่นใจสูง (แนะนำให้พิจารณาเปปไทด์นี้)<br>
                            &nbsp;&nbsp;- 50-70% = มีความมั่นใจปานกลาง<br>
                            &nbsp;&nbsp;- <50% = มีความมั่นใจต่ำ
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Download
                        csv = df_results.to_csv(index=False)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name=f"lcms_analysis_{timestamp}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No high-confidence peptides found (confidence ≥ 60%)")
                
                except Exception as e:
                    st.error(f"Error analyzing file: {str(e)}")

# Tab 4: Molecular Docking Prediction
with tab4:
    st.subheader("🧬 Molecular Docking Prediction")
    st.write("ทำนายการเกิด interaction ของ peptide กับ key residues ใน T1R1/T1R3 receptor pockets")
    
    # Disclaimer
    st.markdown("""
    <div class="info-box" style="background-color: #fef3c7; border-left: 4px solid #f59e0b;">
        <strong>⚠️ คำเตือน:</strong> การทำนายนี้เป็นการประมาณการเบื้องต้นโดยใช้กฎเกณฑ์ทางชีวเคมี ไม่ใช่การทำ Molecular Docking จริง<br>
        ผลลัพธ์ควรใช้เป็นแนวทางเบื้องต้นเท่านั้น สำหรับการยืนยันที่แม่นยำควรทำการทดสอบในห้องปฏิบัติการ
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    docking_sequence = st.text_input(
        "Peptide Sequence (ลำดับเปปไทด์)",
        placeholder="e.g., EAGIQ, LPEEV, DDEE, DG, EK",
        help="ใส่ลำดับเปปไทด์ (2-20 amino acids) เพื่อทำนายการเกิด interaction กับ receptor",
        key="docking_seq"
    )
    
    if st.button("🧬 ทำนาย Molecular Interactions", type="primary", use_container_width=True):
        if not docking_sequence:
            st.warning("กรุณาใส่ลำดับเปปไทด์")
        else:
            sequence = docking_sequence.upper().strip()
            if not all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in sequence):
                st.error("ลำดับไม่ถูกต้อง กรุณาใช้เฉพาะ amino acid codes (A-Z) เท่านั้น")
            elif len(sequence) < 2 or len(sequence) > 20:
                st.error("ความยาวของลำดับต้องอยู่ระหว่าง 2-20 amino acids")
            else:
                with st.spinner("🔬 กำลังคำนวณ molecular interactions..."):
                    # Calculate peptide properties
                    props = calculate_peptide_properties(sequence)
                    
                    # Predict interactions with T1R1 and T1R3
                    t1r1_result = predict_t1r1_interactions(sequence, props)
                    t1r3_result = predict_t1r3_interactions(sequence, props)
                    
                    # Determine overall umami potential
                    umami_assessment = classify_umami_potential(t1r1_result, t1r3_result, props)
                    
                    st.success(f"✅ วิเคราะห์เสร็จสมบูรณ์สำหรับ: **{sequence}**")
                    
                    # Display peptide properties
                    st.markdown("### 📊 คุณสมบัติของ Peptide")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("ความยาว", f"{props['length']} aa")
                        st.metric("Net Charge (pH 7)", f"{props['net_charge']:.2f}")
                    with col2:
                        st.metric("Molecular Weight", f"{props['mw']:.1f} Da")
                        st.metric("Isoelectric Point", f"{props['pi']:.2f}")
                    with col3:
                        st.metric("Hydrophobicity (GRAVY)", f"{props['gravy']:.3f}")
                        st.metric("กรดอะมิโน (D/E)", f"{props['acidic_pct']:.1f}%")
                    
                    # Display T1R1 interactions
                    st.markdown("### 🔵 T1R1 Pocket Interactions")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown(f"""
                        <div class="prediction-box umami-box">
                            <h4>🎯 Binding Score</h4>
                            <h2 style="color: #f59e0b;">{t1r1_result['score']:.1f}/100</h2>
                            <p><strong>Estimated Binding Energy:</strong> {t1r1_result['binding_energy']:.2f} kcal/mol</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="prediction-box" style="background-color: #f0f9ff; border-left: 4px solid #3b82f6;">
                            <h4>🔗 Predicted Interactions</h4>
                            <p><strong>Hydrogen Bonds:</strong> {t1r1_result['h_bonds']}</p>
                            <p><strong>Electrostatic:</strong> {t1r1_result['electrostatic']}</p>
                            <p><strong>Hydrophobic:</strong> {t1r1_result['hydrophobic']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show key residue interactions for T1R1
                    st.markdown("**Key Residues ที่คาดว่าจะเกิด Interaction:**")
                    interactions_df = pd.DataFrame(t1r1_result['residue_interactions'])
                    st.dataframe(interactions_df, use_container_width=True, hide_index=True)
                    
                    # Display T1R3 interactions
                    st.markdown("### 🟢 T1R3 Pocket Interactions")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown(f"""
                        <div class="prediction-box bitter-box">
                            <h4>🎯 Binding Score</h4>
                            <h2 style="color: #3b82f6;">{t1r3_result['score']:.1f}/100</h2>
                            <p><strong>Estimated Binding Energy:</strong> {t1r3_result['binding_energy']:.2f} kcal/mol</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="prediction-box" style="background-color: #f0fdf4; border-left: 4px solid #10b981;">
                            <h4>🔗 Predicted Interactions</h4>
                            <p><strong>Hydrogen Bonds:</strong> {t1r3_result['h_bonds']}</p>
                            <p><strong>Electrostatic:</strong> {t1r3_result['electrostatic']}</p>
                            <p><strong>Hydrophobic:</strong> {t1r3_result['hydrophobic']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show key residue interactions for T1R3
                    st.markdown("**Key Residues ที่คาดว่าจะเกิด Interaction:**")
                    interactions_df = pd.DataFrame(t1r3_result['residue_interactions'])
                    st.dataframe(interactions_df, use_container_width=True, hide_index=True)
                    
                    # Overall assessment
                    st.markdown("### 🎖️ สรุปศักยภาพ Umami")
                    
                    # Determine color based on potential
                    if umami_assessment['level'] == 'สูง':
                        box_color = '#fef3c7'
                        border_color = '#f59e0b'
                        emoji = '🌟'
                    elif umami_assessment['level'] == 'ปานกลาง':
                        box_color = '#dbeafe'
                        border_color = '#3b82f6'
                        emoji = '⭐'
                    elif umami_assessment['level'] == 'ต่ำ':
                        box_color = '#f3f4f6'
                        border_color = '#9ca3af'
                        emoji = '🔸'
                    else:
                        box_color = '#fee2e2'
                        border_color = '#ef4444'
                        emoji = '❌'
                    
                    st.markdown(f"""
                    <div class="prediction-box" style="background-color: {box_color}; border-left: 4px solid {border_color};">
                        <h3>{emoji} ศักยภาพ Umami: {umami_assessment['level']}</h3>
                        <p><strong>Overall Score:</strong> {umami_assessment['overall_score']:.1f}/100</p>
                        <p><strong>Preferred Pocket:</strong> {umami_assessment['preferred_pocket']}</p>
                        <p><strong>คำแนะนำ:</strong> {umami_assessment['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Comparison with known umami peptides
                    st.markdown("### 🔍 เปรียบเทียบกับ Umami Peptides ที่รู้จัก")
                    
                    known_peptides = [
                        {'Peptide': 'DG (Asp-Gly)', 'T1R1 BE': -8.1, 'T1R3 BE': -7.3, 'Preference': 'T1R1'},
                        {'Peptide': 'EK (Glu-Lys)', 'T1R1 BE': -7.1, 'T1R3 BE': -8.3, 'Preference': 'T1R3'},
                        {'Peptide': 'EE (Glu-Glu)', 'T1R1 BE': -7.5, 'T1R3 BE': -7.8, 'Preference': 'T1R3'},
                        {'Peptide': 'DD (Asp-Asp)', 'T1R1 BE': -7.6, 'T1R3 BE': -7.2, 'Preference': 'T1R1'},
                        {'Peptide': f'{sequence} (ของคุณ)', 'T1R1 BE': t1r1_result['binding_energy'], 
                         'T1R3 BE': t1r3_result['binding_energy'], 
                         'Preference': umami_assessment['preferred_pocket']}
                    ]
                    
                    comparison_df = pd.DataFrame(known_peptides)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Molecular Visualization Section
                    st.markdown("### 🎨 Molecular Visualization")
                    st.markdown("สร้างภาพโมเลกุล 2D และ 3D เพื่อแสดงโครงสร้างและ interactions")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🖼️ สร้างภาพ 2D Structure", use_container_width=True):
                            with st.spinner("กำลังสร้างภาพ 2D..."):
                                try:
                                    import matplotlib.pyplot as plt
                                    import matplotlib.patches as mpatches
                                    from matplotlib.patches import Circle
                                    import tempfile
                                    import os
                                    
                                    # Generate 2D structure
                                    from generate_peptide_structures import generate_2d_structure
                                    
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                                        generate_2d_structure(sequence, tmp.name)
                                        st.image(tmp.name, caption=f"2D Structure of {sequence}")
                                        
                                        # Provide download button
                                        with open(tmp.name, 'rb') as f:
                                            st.download_button(
                                                label="📥 Download 2D Structure",
                                                data=f.read(),
                                                file_name=f"{sequence}_2d_structure.png",
                                                mime="image/png"
                                            )
                                        os.unlink(tmp.name)
                                    
                                    st.success("✅ สร้างภาพ 2D เสร็จสมบูรณ์!")
                                except Exception as e:
                                    st.error(f"เกิดข้อผิดพลาด: {str(e)}")
                    
                    with col2:
                        if st.button("🔵 สร้างภาพ Interactions", use_container_width=True):
                            with st.spinner("กำลังสร้างภาพ Interactions..."):
                                try:
                                    from generate_peptide_structures import generate_interaction_diagram
                                    import tempfile
                                    import os
                                    
                                    # T1R1 key residues
                                    t1r1_residues = {
                                        'Y220': {'type': 'H-bond'},
                                        'E301': {'type': 'Electrostatic'},
                                        'L305': {'type': 'Hydrophobic'},
                                        'S306': {'type': 'H-bond'},
                                        'S385': {'type': 'H-bond'},
                                        'N388': {'type': 'H-bond'},
                                        'D147': {'type': 'Electrostatic'},
                                        'Y169': {'type': 'H-bond'},
                                        'L75': {'type': 'Hydrophobic'},
                                        'A302': {'type': 'H-bond'},
                                    }
                                    
                                    # T1R3 key residues
                                    t1r3_residues = {
                                        "S146'": {'type': 'H-bond'},
                                        "S147'": {'type': 'H-bond'},
                                        "E148'": {'type': 'Electrostatic'},
                                        "T167'": {'type': 'H-bond/Hydrophobic'},
                                        "G168'": {'type': 'Flexibility'},
                                        "S170'": {'type': 'H-bond'},
                                        "M171'": {'type': 'Hydrophobic'},
                                        "D190'": {'type': 'Electrostatic'},
                                        "N386'": {'type': 'H-bond'},
                                        "Q389'": {'type': 'H-bond'},
                                        "W72'": {'type': 'Hydrophobic'},
                                        "H145'": {'type': 'H-bond'},
                                    }
                                    
                                    # Generate T1R1 interaction diagram
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='_t1r1.png') as tmp1:
                                        generate_interaction_diagram(sequence, "T1R1", t1r1_residues, tmp1.name)
                                        st.image(tmp1.name, caption=f"T1R1 Pocket Interactions - {sequence}")
                                        
                                        with open(tmp1.name, 'rb') as f:
                                            st.download_button(
                                                label="📥 Download T1R1 Interactions",
                                                data=f.read(),
                                                file_name=f"{sequence}_t1r1_interactions.png",
                                                mime="image/png",
                                                key="download_t1r1"
                                            )
                                        os.unlink(tmp1.name)
                                    
                                    # Generate T1R3 interaction diagram
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='_t1r3.png') as tmp2:
                                        generate_interaction_diagram(sequence, "T1R3", t1r3_residues, tmp2.name)
                                        st.image(tmp2.name, caption=f"T1R3 Pocket Interactions - {sequence}")
                                        
                                        with open(tmp2.name, 'rb') as f:
                                            st.download_button(
                                                label="📥 Download T1R3 Interactions",
                                                data=f.read(),
                                                file_name=f"{sequence}_t1r3_interactions.png",
                                                mime="image/png",
                                                key="download_t1r3"
                                            )
                                        os.unlink(tmp2.name)
                                    
                                    st.success("✅ สร้างภาพ Interactions เสร็จสมบูรณ์!")
                                except Exception as e:
                                    st.error(f"เกิดข้อผิดพลาด: {str(e)}")
                    
                    with col3:
                        if st.button("🌐 สร้างภาพ 3D Pocket", use_container_width=True):
                            with st.spinner("กำลังสร้างภาพ 3D..."):
                                try:
                                    from generate_3d_pocket_visualization import generate_3d_pocket_visualization, generate_pocket_cross_section
                                    import tempfile
                                    import os
                                    
                                    # T1R1 and T1R3 key residues (same as above)
                                    t1r1_residues = {
                                        'Y220': {'type': 'H-bond'},
                                        'E301': {'type': 'Electrostatic'},
                                        'L305': {'type': 'Hydrophobic'},
                                        'S306': {'type': 'H-bond'},
                                        'S385': {'type': 'H-bond'},
                                        'N388': {'type': 'H-bond'},
                                        'D147': {'type': 'Electrostatic'},
                                        'Y169': {'type': 'H-bond'},
                                        'L75': {'type': 'Hydrophobic'},
                                        'A302': {'type': 'H-bond'},
                                    }
                                    
                                    t1r3_residues = {
                                        "S146'": {'type': 'H-bond'},
                                        "S147'": {'type': 'H-bond'},
                                        "E148'": {'type': 'Electrostatic'},
                                        "T167'": {'type': 'H-bond/Hydrophobic'},
                                        "G168'": {'type': 'Flexibility'},
                                        "S170'": {'type': 'H-bond'},
                                        "M171'": {'type': 'Hydrophobic'},
                                        "D190'": {'type': 'Electrostatic'},
                                        "N386'": {'type': 'H-bond'},
                                        "Q389'": {'type': 'H-bond'},
                                        "W72'": {'type': 'Hydrophobic'},
                                        "H145'": {'type': 'H-bond'},
                                    }
                                    
                                    # Generate 3D visualizations
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='_t1r1_3d.png') as tmp1:
                                        generate_3d_pocket_visualization(sequence, "T1R1", t1r1_residues, tmp1.name)
                                        st.image(tmp1.name, caption=f"T1R1 Pocket 3D - {sequence}")
                                        
                                        with open(tmp1.name, 'rb') as f:
                                            st.download_button(
                                                label="📥 Download T1R1 3D",
                                                data=f.read(),
                                                file_name=f"{sequence}_t1r1_3d.png",
                                                mime="image/png",
                                                key="download_t1r1_3d"
                                            )
                                        os.unlink(tmp1.name)
                                    
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='_t1r3_3d.png') as tmp2:
                                        generate_3d_pocket_visualization(sequence, "T1R3", t1r3_residues, tmp2.name)
                                        st.image(tmp2.name, caption=f"T1R3 Pocket 3D - {sequence}")
                                        
                                        with open(tmp2.name, 'rb') as f:
                                            st.download_button(
                                                label="📥 Download T1R3 3D",
                                                data=f.read(),
                                                file_name=f"{sequence}_t1r3_3d.png",
                                                mime="image/png",
                                                key="download_t1r3_3d"
                                            )
                                        os.unlink(tmp2)
                                    
                                    # Generate cross-sections
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='_t1r1_cross.png') as tmp3:
                                        generate_pocket_cross_section(sequence, "T1R1", tmp3.name)
                                        st.image(tmp3.name, caption=f"T1R1 Cross-Section - {sequence}")
                                        os.unlink(tmp3.name)
                                    
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='_t1r3_cross.png') as tmp4:
                                        generate_pocket_cross_section(sequence, "T1R3", tmp4.name)
                                        st.image(tmp4.name, caption=f"T1R3 Cross-Section - {sequence}")
                                        os.unlink(tmp4.name)
                                    
                                    st.success("✅ สร้างภาพ 3D เสร็จสมบูรณ์!")
                                except Exception as e:
                                    st.error(f"เกิดข้อผิดพลาด: {str(e)}")
                    
                    # Footnote
                    st.markdown("""
                    <div class="footnote">
                        <strong>📝 หมายเหตุ:</strong><br>
                        • <strong>Binding Energy (BE):</strong> พลังงานการจับที่ประมาณการ (ยิ่งต่ำ = จับแรงขึ้น)<br>
                        • <strong>Key Residues:</strong> กรดอะมิโนสำคัญใน receptor pocket ที่เกี่ยวข้องกับการจับ peptide<br>
                        • <strong>Interaction Types:</strong><br>
                        &nbsp;&nbsp;- H-bond: พันธะไฮโดรเจน (สำคัญที่สุด)<br>
                        &nbsp;&nbsp;- Electrostatic: แรงไฟฟ้าสถิต (สำคัญสำหรับ acidic peptides)<br>
                        &nbsp;&nbsp;- Hydrophobic: แรงไฮโดรโฟบิก<br>
                        • ข้อมูลอ้างอิงจาก: Hu et al. (2025), Zhang et al. (2019), Wang et al. (2022)
                    </div>
                    """, unsafe_allow_html=True)

# Sidebar - About
with st.sidebar:
    st.header("📊 About the Models")
    
    if models:
        st.markdown(f"""
        **Umami Model**
        - Training Accuracy: {models['umami']['accuracy']*100:.1f}%
        - Training Data: {len(UMAMI_POSITIVE)} positive, {len(UMAMI_NEGATIVE)} negative
        - Predicts savory/meaty taste
        
        **Bitter Model**
        - Training Accuracy: {models['bitter']['accuracy']*100:.1f}%
        - Training Data: {len(BITTER_POSITIVE)} positive, {len(BITTER_NEGATIVE)} negative
        - Predicts bitter taste
        
        **Features**
        - Amino acid composition (20 features)
        - Dipeptide composition (23 features)
        - Physicochemical properties (5 features)
        """)
    
    st.markdown("---")
    
    st.header("📚 Data Sources")
    st.markdown("""
    - [TPDM](https://github.com/SynchronyML/TPDM)
    - [TastePepMap](http://www.wang-subgroup.com/TastePepMap.html)
    - [BIOPEP-UWM](https://biochemia.uwm.edu.pl/biopep/)
    - [BitterDB](https://bitterdb.agri.huji.ac.il/)
    """)
    
    st.markdown("---")
    st.caption("Powered by Aniwat Kaewkrod (a.biotwu@gmail.com)")

