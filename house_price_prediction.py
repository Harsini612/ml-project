# ============================================================
#  House Price Prediction using Linear Regression
# ============================================================

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("  House Price Prediction - Linear Regression")
print("=" * 50)

# -------------------------------------------------------
# STEP 2: Load the Dataset
# -------------------------------------------------------
df = pd.read_csv('homeprices.csv')
print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# -------------------------------------------------------
# STEP 3: Handle Missing Values
# -------------------------------------------------------
# Fill missing bathroom with median
df['bathroom'] = df['bathroom'].fillna(df['bathroom'].median())

# Fill missing age with median
df['age'] = df['age'].fillna(df['age'].median())

print(f"\n✅ Missing values handled.")
print("Remaining nulls:\n", df.isnull().sum())

# -------------------------------------------------------
# STEP 4: Encode Categorical Columns
# (Convert text → numbers so the model can understand)
# -------------------------------------------------------
le = LabelEncoder()

df['status_encoded']   = le.fit_transform(df['status'])
df['location_encoded'] = le.fit_transform(df['location'])
df['builder_encoded']  = le.fit_transform(df['builder'])

print("\n✅ Categorical columns encoded.")

# -------------------------------------------------------
# STEP 5: Select Features (X) and Target (y)
# -------------------------------------------------------
X = df[['area', 'bhk', 'bathroom', 'age',
        'status_encoded', 'location_encoded', 'builder_encoded']]

y = df['price']

print(f"\n✅ Features selected: {list(X.columns)}")
print(f"   Target: price")

# -------------------------------------------------------
# STEP 6: Split into Train and Test sets
# (80% train, 20% test)
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Data split:")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}")

# -------------------------------------------------------
# STEP 7: Train the Linear Regression Model
# -------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Model trained successfully!")

# -------------------------------------------------------
# STEP 8: Make Predictions
# -------------------------------------------------------
y_pred = model.predict(X_test)

# -------------------------------------------------------
# STEP 9: Evaluate the Model
# -------------------------------------------------------
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("  Model Performance")
print("=" * 50)
print(f"  MAE  (Mean Absolute Error) : ₹{mae:.2f} lakhs")
print(f"  RMSE (Root Mean Sq Error)  : ₹{rmse:.2f} lakhs")
print(f"  R²   (Accuracy Score)      : {r2:.4f}  ({r2*100:.1f}%)")
print("=" * 50)

# -------------------------------------------------------
# STEP 10: Show Sample Predictions vs Actual
# -------------------------------------------------------
results = pd.DataFrame({
    'Actual Price (₹L)'   : y_test.values[:10],
    'Predicted Price (₹L)': np.round(y_pred[:10], 2)
})
print("\n📊 Sample Predictions vs Actual:")
print(results.to_string(index=False))

# -------------------------------------------------------
# STEP 11: Predict on a NEW house
# -------------------------------------------------------
print("\n" + "=" * 50)
print("  Predict Price for a NEW House")
print("=" * 50)

# Example: 1200 sqft, 2BHK, 2 bath, 3 yrs old, Ready to move
# For location/builder, we use a numeric code (0 = first in sorted order)
new_house = pd.DataFrame([{
    'area'             : 3000,
    'bhk'              : 4,
    'bathroom'         : 4,
    'age'              : 1,
    'status_encoded'   : 1,   # 1 = Ready to move
    'location_encoded' : 50,  # a mid-range location code
    'builder_encoded'  : 20   # a mid-range builder code
}])

predicted_price = model.predict(new_house)[0]
print(f"\n  Input  : 3000 sqft | 4 BHK | 4 Bath | Age: 1 yrs | Ready to move")
print(f"  💰 Predicted Price: ₹{predicted_price:.2f} lakhs")
print()
