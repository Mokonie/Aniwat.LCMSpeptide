# 🧬 Peptide Taste Predictor

AI-Powered Peptide Taste Analysis from LC-MS/MS Data

**Powered by Aniwat Kaewkrod (a.biotwu@gmail.com)**

---

## 🚀 Deploy to Streamlit Cloud

### Step 1: Upload to GitHub

1. Go to https://github.com/Mokonie/Aniwat.LCMSpeptide
2. Click "Add file" → "Upload files"
3. Drag these files:
   - `app.py`
   - `training_data.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
4. Commit: "Add Peptide Taste Predictor"

### Step 2: Deploy

1. Go to https://share.streamlit.io
2. Click "New app"
3. Repository: `Mokonie/Aniwat.LCMSpeptide`
4. Main file: `app.py`
5. Click "Deploy!"

Done! Your app will be live in 2-5 minutes.

---

## ✨ Features

- **Single Prediction**: Predict taste of one peptide
- **Cut & Analyze**: Cut long peptides into fragments
- **LC-MS/MS Upload**: Batch analysis from CSV files
- **High Accuracy**: ~89% for Umami and Bitter

---

## 📊 Models

- **Umami**: 244 positive + 245 negative samples
- **Bitter**: 233 positive + 256 negative samples
- **Algorithm**: Random Forest (100 trees)
- **Features**: AA composition, dipeptides, physicochemical properties

---

## 🛠️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open: http://localhost:8501

---

## 📚 Data Sources

- TPDM: https://github.com/SynchronyML/TPDM
- TastePepMap: http://www.wang-subgroup.com/TastePepMap.html
- BIOPEP-UWM: https://biochemia.uwm.edu.pl/biopep/
- BitterDB: https://bitterdb.agri.huji.ac.il/

---

**Note**: This tool is for research purposes only.
