# Import required libraries
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from prophet import Prophet  # Import Prophet from the newer library
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the cleaned dataset
data = pd.read_excel("Cleaned_Cisco_Forecast_Data.xlsx")

# Filter relevant columns
columns = [
    "Cost Rank", "Product Name",
    "FY22 Q2", "FY22 Q3", "FY22 Q4", "FY23 Q1", "FY23 Q2", "FY23 Q3",
    "FY23 Q4", "FY24 Q1", "FY24 Q2", "FY24 Q3", "FY24 Q4", "FY25 Q1"
]
data = data[columns]

# Function to prepare data for XGBoost with rolling windows
def prepare_xgboost_data(product_data):
    features = []
    targets = []
    
    # Rolling window approach: Use 4 previous quarters to predict next quarter
    for i in range(len(product_data) - 4):
        features.append(product_data[i:i+4])
        targets.append(product_data[i+4])
    
    return np.array(features), np.array(targets)

# Function to prepare data for Prophet
def prepare_prophet_data(product_data, product_name):
    prophet_data = pd.DataFrame({
        'ds': pd.date_range(start='2022-08-01', periods=len(product_data.columns[2:]), freq='QE'),
        'y': product_data.iloc[0, 2:].values
    })
    prophet_data["product"] = product_name  # Add product grouping
    return prophet_data

# Dictionary to store predictions
predictions = {
    "Product Name": [],
    "Cost Rank": [],
    "XGBoost Prediction": [],
    "Prophet Prediction": []
}

# Iterate over each product
for index, row in data.iterrows():
    try:
        product_name = row["Product Name"]
        cost_rank = row["Cost Rank"]
        
        # Skip products with insufficient data (e.g., all zeros)
        if (row.iloc[2:] == 0).all():
            print(f"Skipping {product_name} due to insufficient data.")
            continue
        
        # Prepare data for XGBoost
        features, target = prepare_xgboost_data(row.iloc[2:].values)
        
        if len(features) == 0:  # Skip if not enough data
            print(f"Skipping {product_name} due to insufficient data for XGBoost.")
            continue
        
        # Train-test split (80% training, 20% validation)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Train XGBoost model with regularization
        xgb_model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,  # Reduce to prevent overfitting
            max_depth=4,  # Reduce complexity
            learning_rate=0.01,  # Increase learning rate slightly
            subsample=0.8,  # Use only 80% of the data per tree
            colsample_bytree=0.8,  # Use only 80% of the features per tree
            early_stopping_rounds=10,
            random_state=42
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # Predict for next quarter (FY25 Q2)
        xgb_pred = xgb_model.predict(features[-1].reshape(1, -1))[0]
        
        # Prepare data for Prophet (grouped training)
        prophet_data = prepare_prophet_data(row.to_frame().T, product_name)
        
        # Train Prophet model
        prophet_model = Prophet()
        prophet_model.fit(prophet_data)
        
        # Create future dataframe for prediction
        future = prophet_model.make_future_dataframe(periods=1, freq='QE')
        forecast = prophet_model.predict(future)
        prophet_pred = forecast['yhat'].iloc[-1]  # Predict for FY25 Q2
        
        # Store predictions
        predictions["Product Name"].append(product_name)
        predictions["Cost Rank"].append(cost_rank)
        predictions["XGBoost Prediction"].append(round(xgb_pred, 2))
        predictions["Prophet Prediction"].append(round(prophet_pred, 2))
        
        print(f"Predictions for {product_name}:")
        print(f"  Cost Rank: {cost_rank}")
        print(f"  XGBoost Prediction: {xgb_pred:.2f}")
        print(f"  Prophet Prediction: {prophet_pred:.2f}")
        print("-" * 40)
    
    except Exception as e:
        print(f"Error processing {product_name}: {e}")

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions)

# Save predictions to Excel
predictions_df.to_excel("XGBoost_vs_Prophet_Predictions_Fixed.xlsx", index=False)

print("Predictions saved to 'XGBoost_vs_Prophet_Predictions_Fixed.xlsx'")
