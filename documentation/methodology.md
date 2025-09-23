# Analytical Methodology

## Problem Framing
- **Task**: Multi-class classification of case resolution status
- **Classes**: Unsolved (0), Solved by Arrest (1), Solved Exceptionally (2)
- **Challenge**: Severe class imbalance (80% vs 10% vs 10%)

## Data Preprocessing

### Handling Missing Data
- Weapon Used Cd: Removed due to 67% missingness and data leakage risk
- Other features: Median imputation for numerical, mode for categorical

### Feature Engineering
- **Temporal Features**: time_to_report, time bins, day of week
- **Geographic Features**: K-means clustering on LAT/LON coordinates  
- **MO Code Processing**: Embedding layer to learn semantic code relationships
- **Crime Complexity**: crime_count from multiple crime code fields

## Modeling Approach

### Algorithms Tested
1. **Neural Network**: Failed - predicted only majority class
2. **XGBoost (Tuned)**: Good performance but slightly over-regularized  
3. **XGBoost (Standard)**: Selected - best balance of precision/recall

### Class Imbalance Handling
- Class weighting (inverse proportional to frequency)
- Careful metric selection (F1-score, recall for minority classes)

## Evaluation Strategy
- **Primary Metric**: Macro F1-score (balance across all classes)
- **Secondary Metrics**: Class-wise precision and recall
- **Validation**: Stratified 80/20 train-test split

## Why XGBoost Was Selected
- Successfully learned all three classes (unlike neural network)
- Best overall F1-score performance  
- More interpretable through feature importance
- Better handling of class imbalance