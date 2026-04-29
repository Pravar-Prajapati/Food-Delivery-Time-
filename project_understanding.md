# Food Delivery Time Prediction - Project Notes

## Goal

Predict `Delivery_Time_min` from order, weather, traffic, time-of-day, vehicle, preparation time, and courier experience features.

This is a regression problem, so the main metrics are:

- `R^2` for explained variance
- `RMSE` for prediction error in minutes
- `MAE` for average absolute error in minutes

## Dataset Summary

- Source file: `Food_Delivery_Times.csv`
- Target column: `Delivery_Time_min`
- Identifier removed: `Order_ID`
- Categorical features used:
  - `Weather`
  - `Traffic_Level`
  - `Time_of_Day`
  - `Vehicle_Type`

## Preprocessing Used

- Missing categorical values filled with the mode
- Missing `Courier_Experience_yrs` filled with the median
- Baseline models used label encoding and scaling
- Stronger model used one-hot encoding for categorical variables
- Added interaction features for tree-based models:
  - `distance_traffic`
  - `distance_prep`
  - `experience_traffic`

## Models Tested

Baseline models tested in the notebook and scripts:

- Linear Regression
- Ridge
- Lasso
- ElasticNet
- SGD Regressor
- Extra Tree Regressor
- Gradient Boosting Regressor
- Hist Gradient Boosting Regressor
- Random Forest Regressor
- Extra Trees Regressor
- AdaBoost Regressor
- KNeighbors Regressor
- Decision Tree Regressor
- SVR
- MLP Regressor
- XGBRegressor when available

## Best Model Found

The best validated model is:

- OneHot encoding + stacked ensemble + target transformation

Final holdout performance:

- `R^2 = 0.8283`
- `RMSE = 8.7727`
- `MAE = 5.8996`

This crossed the target of `R^2 > 0.82`.

## Why It Worked Better

- One-hot encoding kept the categorical information more faithfully than label encoding.
- Stacking combined multiple tree-based learners instead of relying on one model.
- Log target transformation helped stabilize skew in delivery-time values.
- Interaction features gave the model better signal for traffic, distance, and courier experience relationships.

## Files Created

- `compute_model_metrics.py` - compares baseline and advanced models
- `improved_model_metrics.csv` - earlier tuning results
- `advanced_model_metrics.csv` - advanced comparison results
- `best_robust_model.pkl` - saved best model pipeline

## How To Re-run

```bash
python compute_model_metrics.py
```

## Suggested Next Steps

1. Add cross-validation reporting for the best model only.
2. Try more feature engineering around traffic, time of day, and vehicle type.
3. Save a small inference script for predicting new delivery times.
4. Mirror the final pipeline inside the notebook for presentation.