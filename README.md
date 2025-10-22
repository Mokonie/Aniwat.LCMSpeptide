# 🧬 Peptide Taste Predictor (Streamlit)

AI-Powered Peptide Taste Analysis from LC-MS/MS Data

**Powered by Aniwat Kaewkrod (a.biotwu@gmail.com)**

---

## 🚀 Quick Start

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud

### Step 1: Push to GitHub

1. Create a new repository on GitHub
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `models/` folder (with all .pkl files)
   - `data/` folder (with all .csv files)

### Step 2: Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your GitHub repository
4. Set main file path: `app.py`
5. Click "Deploy"

---

## 📁 Project Structure

```
peptide-taste-predictor-streamlit/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── models/                # Trained ML models
│   ├── umami_model.pkl
│   ├── umami_features.pkl
│   ├── bitter_model.pkl
│   └── bitter_features.pkl
└── data/                  # Training datasets (optional)
    ├── peptides_umami.csv
    ├── peptides_bitter.csv
    └── peptides_sweet.csv
```

---

## ✨ Features

### 1. Single Prediction
- Enter a peptide sequence (2-20 amino acids)
- Get instant taste predictions
- Umami and Bitter predictions with ~89% accuracy

### 2. Cut & Analyze
- Input long peptide sequences
- Automatically cut into fragments
- Predict taste for all fragments
- Download results as CSV

### 3. LC-MS/MS File Analysis
- Upload CSV files from LC-MS/MS
- Supports two formats:
  - `peptidefeatures.csv`
  - `alldenovocandidates.csv`
- Filter high-confidence peptides
- Batch prediction and export

---

## 🧬 ML Models

### Umami Model
- **Accuracy**: 89.8%
- **Training Data**: 489 peptides
- **Predicts**: Savory/meaty taste

### Bitter Model
- **Accuracy**: 88.8%
- **Training Data**: 489 peptides
- **Predicts**: Bitter taste

### Features Used
- Amino acid composition (20 features)
- Dipeptide composition (23 features)
- Physicochemical properties (5 features)
  - Hydrophobicity
  - Charge
  - Polarity
  - Aromaticity
  - Length

---

## 📊 Data Sources

- **TPDM**: https://github.com/SynchronyML/TPDM
- **TastePepMap**: http://www.wang-subgroup.com/TastePepMap.html
- **BIOPEP-UWM**: https://biochemia.uwm.edu.pl/biopep/
- **BitterDB**: https://bitterdb.agri.huji.ac.il/

---

## 🛠️ Requirements

- Python 3.8+
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- joblib >= 1.3.0

---

## 📝 Usage Examples

### Example 1: Single Prediction
```
Input: EAGIQ
Output:
  - Umami: Yes (88.8%)
  - Bitter: Yes (91.9%)
```

### Example 2: Cut & Analyze
```
Input: DDDEEEEEEEEEEEK
Output: 19 fragments
Top fragments:
  1. DDDE - Umami: 96.8%, Bitter: 95.5%
  2. EEEE - Umami: 96.7%, Bitter: 91.6%
  3. DDE - Umami: 96.6%, Bitter: 95.9%
```

### Example 3: LC-MS/MS File
```
Input: alldenovocandidates.csv
Output:
  - Total peptides: 100
  - High confidence (>80%): 45
  - Top peptides with taste predictions
```

---

## 🔧 Troubleshooting

### Models not loading
- Make sure `models/` folder contains all 4 .pkl files
- Check file permissions

### File upload fails
- Maximum file size: 200MB
- Supported format: CSV only

### Invalid sequence error
- Use only single-letter amino acid codes (A-Z)
- Length must be 2-20 amino acids for single prediction

---

## 📄 License

Apache License 2.0

---

## 📧 Contact

**Aniwat Kaewkrod**
- Email: a.biotwu@gmail.com
- GitHub: https://github.com/Mokonie/Aniwat.LCMSpeptide

---

## 🙏 Acknowledgments

This tool uses data from multiple public databases and research publications. 
Please cite the original sources when using this tool in your research.

---

**Note**: This tool is for research purposes only. Taste predictions should be validated through sensory evaluation before use in product development.

