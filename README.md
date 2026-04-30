# 🍔 Food Delivery Time Prediction

A machine learning project to predict food delivery times based on various factors like distance, weather, traffic, and courier experience.

---

## 📊 Project Overview

| Attribute | Details |
|-----------|---------|
| **Dataset** | Food_Delivery_Times.csv |
| **Target Variable** | Delivery_Time_min |
| **Best Model** | Linear Regression / Ridge |
| **R² Score** | ~0.77 (Cross-Validated) |

---

## 🚀 Features

### Input Features
- **Weather**: Clear, Cloudy, Rainy, Stormy
- **Traffic_Level**: Low, Medium, High
- **Time_of_Day**: Morning, Afternoon, Evening, Night
- **Vehicle_Type**: Bicycle, Bike, Car
- **Distance_km**: Delivery distance in kilometers
- **Preparation_Time_min**: Food preparation time
- **Courier_Experience_yrs**: Courier's experience in years

### Output
- **Delivery_Time_min**: Predicted delivery time in minutes

---

## 📁 Project Structure

```
├── time-prediction.ipynb      # Main Jupyter notebook
├── Food_Delivery_Times.csv    # Dataset
├── best_robust_model_final.pkl # Trained model
├── Test_CV.md                 # Cross-validation documentation
├── project_understanding.md   # Project details
├── RESEARCH_DOCUMENTATION.md  # Research findings
└── guide.md                   # User guide
```

---

## 🔬 Methodology

### Phase 1: Data Loading & Exploration
- Load dataset with pandas
- Explore data shape, columns, and types
- Check for missing values

### Phase 2: Data Cleaning & EDA
- Remove unnecessary columns (Order_ID)
- Fill missing values:
  - Categorical: Mode imputation
  - Numerical: Median imputation
- Visualizations:
  - Correlation heatmap
  - Scatter plots (Distance vs Delivery Time)
  - Box plots (Weather, Traffic, Vehicle, Time of Day impact)

### Phase 3: Data Preprocessing
- Define features (X) and target (y)
- Identify categorical and numerical columns
- Train/Test split (80/20)
- Create preprocessor:
  - OneHotEncoder for categorical features
  - MinMaxScaler for numerical features

### Phase 4: Model Comparison
Compare 14 different regression models:

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.8262 | - | - |
| Ridge | 0.8262 | - | - |
| MLP | 0.8237 | - | - |
| CatBoost | 0.8102 | - | - |
| Gradient Boosting | 0.8086 | - | - |
| Random Forest | 0.7897 | - | - |
| Extra Trees | 0.7712 | - | - |
| XGBoost | 0.7564 | - | - |

### Phase 5: Cross-Validation Testing
- **5B**: Train-Test Gap Analysis to detect overfitting
- **5C**: Hyperparameter tuning for Ridge Regression
- **5D**: Hyperparameter tuning for MLP (Multi-Layer Perceptron)
- **5E**: Compare Original Test R² vs Cross-Validation R²
- **5F**: Leave-One-Out Cross-Validation (LOOCV) for robustness

### Phase 6: Feature Engineering
- Create new features:
  - Interaction terms (e.g., Distance × Preparation Time)
  - Flags for rush hour, bad weather, and high traffic
  - Experience-to-distance ratio
- Rebuild preprocessor and retrain models

### Phase 7: Final Model Evaluation
- Test top models (Linear Regression, Ridge, MLP) on engineered features
- Perform new cross-validation tests
- Select the best model for deployment

---

## 📈 Key Results

### Best Performing Models (by CV R²)

| Rank | Model | CV Mean R² | CV Std |
|------|-------|------------|--------|
| 1 | Linear Regression | 0.7685 | 0.0368 |
| 2 | Ridge | 0.7685 | 0.0367 |
| 3 | MLP | 0.7614 | 0.0343 |
| 4 | Gradient Boosting | 0.7371 | 0.0320 |
| 5 | CatBoost | 0.7351 | 0.0300 |

### Overfitting Analysis

| Model | Train R² | Test R² | Gap | Status |
|-------|----------|---------|-----|--------|
| Linear Regression | 0.7630 | 0.8262 | -0.0632 | ✅ OK |
| Ridge | 0.7629 | 0.8262 | -0.0634 | ✅ OK |
| CatBoost | 0.9618 | 0.8102 | 0.1517 | ⚠️ Overfit |
| Random Forest | 0.9551 | 0.7897 | 0.1654 | ⚠️ Overfit |
| XGBoost | 0.9988 | 0.7564 | 0.2424 | ⚠️ Overfit |

---

## 🏆 Conclusion

- **Best Model**: Linear Regression or Ridge (CV R² = 0.7685)
- **True Generalization R²**: ~0.77
- **Simple models outperform complex ones** in this dataset
- **Tree-based models show overfitting** - avoid for production

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/sujalsolanki125/Food-Delivery-Time-Prediction.git

# Navigate to the project
cd Food-Delivery-Time-Prediction

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost
```

---

## 📝 Usage

1. Open `time-prediction.ipynb` in Jupyter Notebook
2. Run cells sequentially
3. The model will be trained and evaluated automatically
4. Use `best_robust_model_final.pkl` for predictions

```python
import pickle

# Load the model
model = pickle.load(open("best_robust_model_final.pkl", "rb"))

# Make predictions
predictions = model.predict(X_new)
```

---

## 📚 Documentation

- [Test_CV.md](Test_CV.md) - Detailed cross-validation testing
- [project_understanding.md](project_understanding.md) - Project overview
- [RESEARCH_DOCUMENTATION.md](RESEARCH_DOCUMENTATION.md) - Research findings
- [guide.md](guide.md) - User guide

---

## 👤 Author

**Sujal Solanki**

---

## 📅 Date

April 30, 2026

---

## 🔗 Links

- [GitHub Repository](https://github.com/sujalsolanki125/Food-Delivery-Time-Prediction)
- [Dataset](Food_Delivery_Times.csv)
- [Model File](best_robust_model_final.pkl)