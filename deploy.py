import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the GOOG stock data for model training
df = pd.read_csv('GOOG.csv')

# Convert the date column to datetime for time series analysis
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Feature Selection
features = ['open', 'high', 'low', 'volume', 'adjClose', 'adjHigh', 'adjLow', 'adjOpen']
target = 'close'

# Split data into features (X) and target (y)
X = df[features]
y = df[target]

# Normalize the feature data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

lr_model = LinearRegression()
lr_model.fit(X_scaled, y)

# Streamlit app title
st.title("Stock Price Prediction")

# Instructions for user input
st.subheader("Enter stock information to predict the closing price:")

# Create form to take user input for features
with st.form("prediction_form"):
    open_val = st.number_input("Opening Price", min_value=0.0, step=0.01)
    high_val = st.number_input("Highest Price", min_value=0.0, step=0.01)
    low_val = st.number_input("Lowest Price", min_value=0.0, step=0.01)
    volume_val = st.number_input("Volume", min_value=0, step=1)
    adj_close_val = st.number_input("Adjusted Close Price", min_value=0.0, step=0.01)
    adj_high_val = st.number_input("Adjusted High Price", min_value=0.0, step=0.01)
    adj_low_val = st.number_input("Adjusted Low Price", min_value=0.0, step=0.01)
    adj_open_val = st.number_input("Adjusted Opening Price", min_value=0.0, step=0.01)
    
    # Model selection
    model_choice = st.selectbox("Choose Regression Model", ("Random Forest", "Linear Regression"))
    
    # Submit button
    submitted = st.form_submit_button("Predict")

# If the user submits the form, perform the prediction
if submitted:
    # Scale the input data
    user_input = np.array([[open_val, high_val, low_val, volume_val, adj_close_val, adj_high_val, adj_low_val, adj_open_val]])
    user_input_scaled = scaler.transform(user_input)
    
    # Predict based on chosen model
    if model_choice == "Random Forest":
        prediction = rf_model.predict(user_input_scaled)[0]
    elif model_choice == "Linear Regression":
        prediction = lr_model.predict(user_input_scaled)[0]
    
    # Display the predicted closing price
    st.subheader(f"Predicted Closing Price: ${prediction:.2f}")
