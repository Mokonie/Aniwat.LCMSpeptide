# Peptide Taste Predictor

AI-Powered Peptide Taste Analysis from LC-MS/MS Data

## 🎯 Overview

This web application uses Machine Learning models to predict taste properties (Umami, Bitter, Sweet) of peptide sequences derived from LC-MS/MS analysis. It's designed for researchers working with protein hydrolysates, bioactive peptides, and food science applications.

## ✨ Features

### 1. **Single Peptide Prediction**
- Enter a peptide sequence (2-20 amino acids)
- Get instant taste predictions with confidence scores
- Supports Umami and Bitter taste prediction (~89% accuracy)

### 2. **Cut & Analyze**
- Input long peptide sequences (up to 100 amino acids)
- Automatically cuts into fragments using sliding window (2-5 AA)
- Predicts taste for all fragments
- Ranks fragments by highest taste confidence

### 3. **LC-MS/MS File Analysis**
- Upload CSV files from LC-MS/MS analysis
- Supports two file types:
  - `peptidefeatures.csv` - Contains peptide sequences with confidence scores
  - `alldenovocandidates.csv` - Contains de novo sequencing results
- Filters high-confidence peptides (>80% confidence)
- Displays top peptides for further analysis

## 🧬 Machine Learning Models

### Umami Model
- **Training Data**: 489 peptides (244 positive, 245 negative)
- **Accuracy**: 89.8%
- **Features**: Amino acid composition, dipeptide composition, physicochemical properties
- **Key Indicators**: High polarity, negative charge (E, D amino acids)

### Bitter Model
- **Training Data**: 489 peptides (233 positive, 256 negative)
- **Accuracy**: 88.8%
- **Features**: Amino acid composition, dipeptide composition, physicochemical properties
- **Key Indicators**: Hydrophobicity, aromaticity (F, W, Y amino acids)

### Sweet Model
- **Status**: Limited data (17 peptides)
- **Note**: Currently returns N/A - requires more training data

## 🔬 Technical Details

### Feature Engineering

The models use 48 features extracted from peptide sequences:

1. **Amino Acid Composition (20 features)**
   - Frequency of each amino acid (A-Z)

2. **Dipeptide Composition (23 features)**
   - Frequency of common dipeptides (AA, AE, DD, DE, EE, etc.)

3. **Physicochemical Properties (5 features)**
   - Average hydrophobicity
   - Average charge
   - Average polarity
   - Average aromaticity
   - Sequence length

### Algorithm

- **Model Type**: Random Forest Classifier
- **Parameters**:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 5
  - min_samples_leaf: 2

## 📊 Data Sources

The training data was collected from:

1. **TPDM GitHub Repository** - 489 peptides (Umami + Bitter)
2. **TastePepMap Database** - 17 sweet peptides
3. **BIOPEP-UWM** - Bioactive peptide database
4. **BitterDB** - Bitter compound database

## 🚀 Usage Examples

### Example 1: Single Prediction

```
Input: EAGIQ
Output:
  - Umami: Yes (88.8% confidence)
  - Bitter: Yes (91.9% confidence)
```

### Example 2: Cut & Analyze

```
Input: DDDEEEEEEEEEEEK
Output: 19 fragments generated
Top fragments:
  1. DDDE - Umami: 96.8%, Bitter: 95.5%
  2. EEEE - Umami: 96.7%, Bitter: 91.6%
  3. DDE - Umami: 96.6%, Bitter: 95.9%
```

### Example 3: LC-MS/MS File

```
Input: peptidefeatures.csv (6 peptides)
Output:
  - Total peptides: 6
  - High confidence (>80%): 6
  - Top peptides: DDDEEEEEEEEEEEK, DDEEEEEEEEEEEK, etc.
```

## 📁 Project Structure

```
peptide-taste-predictor/
├── client/                 # Frontend (React + TypeScript)
│   └── src/
│       ├── pages/
│       │   └── Home.tsx   # Main UI
│       └── lib/
│           └── trpc.ts    # API client
├── server/                # Backend (Node.js + tRPC)
│   └── routers.ts        # API endpoints
├── data/                  # Training data
│   ├── peptides_umami.csv
│   ├── peptides_bitter.csv
│   └── peptides_sweet.csv
├── models/                # Trained ML models
│   ├── umami_model.pkl
│   ├── bitter_model.pkl
│   ├── umami_features.pkl
│   └── bitter_features.pkl
├── train_models.py       # Model training script
└── predict_taste.py      # Prediction script
```

## 🛠️ Development

### Prerequisites

- Node.js 22+
- Python 3.11+
- pnpm

### Installation

```bash
# Install dependencies
pnpm install

# Train models (if needed)
python3.11 train_models.py

# Start development server
pnpm dev
```

### API Endpoints

#### 1. Predict Single Peptide
```typescript
trpc.peptide.predict.useMutation({
  sequence: "EAGIQ"
})
```

#### 2. Cut and Predict
```typescript
trpc.peptide.cutAndPredict.useMutation({
  sequence: "DDDEEEEEEEEEEEK"
})
```

#### 3. Analyze File
```typescript
trpc.peptide.analyzeFile.useMutation({
  fileContent: "...",
  fileType: "peptidefeatures"
})
```

## 📈 Model Performance

### Cross-Validation Results

**Umami Model:**
- CV Accuracy: 85.9% (±3.5%)
- Test Accuracy: 89.8%
- Precision: 88%
- Recall: 92%

**Bitter Model:**
- CV Accuracy: 87.7% (±4.2%)
- Test Accuracy: 88.8%
- Precision: 88%
- Recall: 89%

### Feature Importance

**Top 5 Features (Umami):**
1. Average polarity (13.7%)
2. Average charge (11.9%)
3. AAC_E (11.6%)
4. Average hydrophobicity (7.6%)
5. AAC_D (7.4%)

**Top 5 Features (Bitter):**
1. Average polarity (12.9%)
2. Average charge (10.9%)
3. AAC_E (10.7%)
4. Average hydrophobicity (8.0%)
5. Average aromaticity (7.1%)

## 🔮 Future Improvements

1. **Expand Training Data**
   - Collect more Sweet peptide data
   - Add Salty and Sour taste data
   - Include more diverse peptide sources

2. **Model Enhancement**
   - Try deep learning models (LSTM, Transformer)
   - Add ensemble methods
   - Implement active learning

3. **Features**
   - Add batch processing for multiple files
   - Export results to CSV/Excel
   - Add visualization of peptide properties
   - Implement peptide similarity search

4. **Integration**
   - Connect to protein databases (UniProt, PDB)
   - Add molecular docking predictions
   - Integrate with sensory evaluation data

## 📚 References

1. **TastePeptides-Meta Platform** - http://www.tastepeptides-meta.com/
2. **BIOPEP-UWM Database** - https://biochemia.uwm.edu.pl/biopep/
3. **BitterDB** - https://bitterdb.agri.huji.ac.il/
4. **Random Forest for Peptide Classification** - Scikit-learn Documentation

## 📝 Citation

If you use this tool in your research, please cite:

```
Peptide Taste Predictor (2025)
AI-Powered Peptide Taste Analysis from LC-MS/MS Data
https://github.com/your-repo/peptide-taste-predictor
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This tool is for research purposes only. Taste predictions should be validated through sensory evaluation before use in product development.

