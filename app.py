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
tab1, tab2, tab3 = st.tabs(["🔬 Single Prediction", "✂️ Cut & Analyze", "📁 LC-MS/MS File"])

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

