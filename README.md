# Crime Case Resolution Prediction Model

## 📋 Project Overview

This machine learning project predicts the likelihood of a crime case being solved based on Los Angeles Police Department data from 2020-2025. The model classifies cases into three categories: **Unsolved**, **Solved by Arrest**, and **Solved Exceptionally**.

## 🏢 Dataset

**Source:** Los Angeles Police Department (LAPD)  
**Time Period:** 2020-2025  
**Records:** ~1 million crime incidents  
**Location:** Los Angeles, California
**Description**: This dataset contains detailed records of reported crimes in Los Angeles, including type of crime, date and time, location, and other relevant attributes. Due to the large size of the data , it cannot be uploaded directly to GitHub. You can download the full dataset directly from the official source: Crime Data from 2020 to Present(https://catalog.data.gov/dataset/crime-data-from-2020-to-present)
.
## 📁 Project Structure

```
Case-Clearance-Prediction-/
├── 📄 CrimeSolvedPrediction.ipynb.  # Main analysis notebook
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # Project documentation
├── 📄 .gitignore                    # Git ignore rules
├── 📂 data/                         # All data files
│   ├── 📄 Old_Crime_Data.csv        # Original raw data
│   └── 📄 processed_data.csv        # Cleaned & engineered data
├── 📂 models/                       # All trained models
│   ├── 📄 crime_prediction_model.pkl # XGBoost model (selected)
│   └── 📄 neural_network_model.keras # Neural network model
└── 📂 documentation/                # Additional docs
    ├── 📄 data_dictionary.md        # Column descriptions
    └── 📄 methodology.md            # Approach details
    └── 📄 Report.pdf
```

## 🚀 Usage

Clone the repository:
```bash
git clone https://github.com/mariammouh/Case-Clearance-Prediction-.git
```
cd Case-Clearance-Prediction-
**Then simply execute the Jupyter notebook cells in sequential order:**

1. **Open** `CrimeSolvedPrediction.ipynb` in Jupyter Notebook
2. **Run all cells sequentially** from top to bottom
3. **Follow the logical flow** through each section:
   - Data loading and inspection
   - Data cleaning and preprocessing  
   - Feature engineering
   - Model training (XGBoost + Neural Network)
   - Model evaluation and comparison
   - Results analysis

The notebook is designed to run completely from start to finish with no additional setup required.

## 🎯 Results

**Selected Model:** XGBoost Classifier  
**Performance:**
- Unsolved cases: 95% precision, 74% recall
- Solved by Arrest: 28% precision, 49% recall  
- Solved Exceptionally: 35% precision, 69% recall

The XGBoost model was selected over the neural network approach, which failed to learn minority classes.

## 📦 Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

## 🔧 Technical Details

- **Primary Algorithm:** XGBoost with class weighting
- **Feature Engineering:** MO code embeddings, temporal features, geographic clustering
- **Validation:** Stratified train-test split with careful evaluation metrics

## 📝 License

This project uses publicly available LAPD data for research and educational purposes.

---

**Note:** Simply run the notebook cells in order to reproduce the complete analysis from raw data to final model selection.