# Cisco Sales Forecasting - FY'25 | Data Pioneers

This project was built for the **Cisco Forecast League FY'25** where my team **Data Pioneers** forecasted quarterly sales for real-life Cisco products using machine learning and time series modeling.

🏆 **Achievements:**
- 📊 Accuracy: **87.47%**
- 🥉 Ranked **15th out of 43** in MIT Bengaluru
- 🏅 Ranked **98th out of 409** across Bangalore

## 📌 Objective
Forecast FY25 Q2 sales for Cisco products based on historical quarterly data using:
- 📈 XGBoost (supervised learning with rolling windows)
- 📉 Prophet (additive time series forecasting model)

## 🧠 Approach
- Data preprocessing and feature engineering
- Supervised rolling window approach for XGBoost
- Time series modeling using Prophet
- Cross-validation using train-test split
- Final predictions compared across models

## 📁 Repository Structure
- `data/`: Excel datasets (anonymized if needed)
- `src/forecast.py`: Core script to generate forecasts
- `outputs/`: Final predictions
- `notebooks/`: Jupyter version for EDA or experimentation

## 🔧 Technologies
- Python
- Pandas, NumPy
- XGBoost
- Prophet
- Scikit-learn

## 📈 Sample Output
![Sample chart or table showing XGBoost vs Prophet predictions]

## 🚀 How to Run
```bash
pip install -r requirements.txt
python src/forecast.py
