  # House Price Prediction 🏠

A machine learning project to predict house prices in Chennai using Linear Regression.

## Dataset
The dataset contains the following features:
- Area (sq ft)
- Number of BHK
- Number of bathrooms
- Age of the property
- Location
- Builder
- Status (ready to move / under construction)

## Model
- Algorithm: Linear Regression
- Library: scikit-learn
- Current Accuracy: 65.6% (R² score)

## Files
- `homeprices.csv` — Dataset used for training and testing
- `house_price_prediction.py` — Main Python script for the model

## Requirements
pip install pandas numpy scikit-learn

## How to Run
python house_price_prediction.py

## Next Steps
- [ ] Try XGBoost for better accuracy
- [ ] Plot actual vs predicted price graphs
- [ ] Build a simple UI for predictions
