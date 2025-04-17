# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 19:09:10 2025

@author: Udayan
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
import sklearn.datasets
import time

# Page configuration
st.set_page_config(page_title="House Price Predictor", layout="wide", page_icon="🏠")

# Title and description
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #2E7D32;'>🏠 California House Price Prediction App</h1>
        <p style='font-size: 18px; color: #333;'>Use real-time inputs to predict housing prices and explore meaningful insights through interactive visualizations.</p>
    </div>
""", unsafe_allow_html=True)

# Load the dataset
house_price_dataset = sklearn.datasets.fetch_california_housing()
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)
house_price_dataframe['Price'] = house_price_dataset.target

# Sidebar for user input
st.sidebar.header("🔧 Customize Your Property")

def user_input_features():
    MedInc = st.sidebar.slider('Median Income', 0.5, 15.0, 3.0)
    HouseAge = st.sidebar.slider('House Age', 1, 52, 20)
    AveRooms = st.sidebar.slider('Average Rooms', 0.8, 141.9, 5.0)
    AveBedrms = st.sidebar.slider('Average Bedrooms', 0.3, 34.0, 1.0)
    Population = st.sidebar.slider('Population', 3, 35682, 1000)
    AveOccup = st.sidebar.slider('Average Occupants', 0.5, 1243.0, 3.0)
    Latitude = st.sidebar.slider('Latitude', 32.5, 42.0, 36.0)
    Longitude = st.sidebar.slider('Longitude', -124.0, -114.0, -120.0)
    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Split the data
X = house_price_dataframe.drop(['Price'], axis=1)
Y = house_price_dataframe['Price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
model = XGBRegressor()
model.fit(X_train, Y_train)

# Make prediction
with st.spinner("Predicting house price..."):
    time.sleep(1)
    prediction = model.predict(input_df)[0] * 100000

# Display major output without animation
st.markdown(f"""
    <div style='background: #f1f8e9; padding: 30px; border-radius: 15px; margin: 20px 0; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);'>
        <h2 style='color: #2E7D32;'>🏡 Estimated House Price</h2>
        <h1 style='font-size: 48px; color: #1B5E20;'>${prediction:,.2f}</h1>
    </div>
""", unsafe_allow_html=True)

# Model Predictions for Evaluation
training_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

# Correlation Heatmap
st.markdown("## 🔍 Feature Correlation Heatmap")
correlation = house_price_dataframe.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', linewidths=1, linecolor='white', ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title("Correlation Between Features", fontsize=14, color='black')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color('black')
st.pyplot(fig)

# Actual vs Predicted Scatter Plot with improved style
st.markdown("## 📈 Actual vs Predicted Prices")
fig2, ax2 = plt.subplots()
sns.scatterplot(x=Y_test, y=test_prediction, ax=ax2, color='aqua', edgecolor='black', s=50, alpha=0.7)
ax2.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
ax2.set_xlabel("Actual Price", color='black')
ax2.set_ylabel("Predicted Price", color='black')
ax2.grid(True, linestyle='--', alpha=0.6)
fig2.patch.set_facecolor('white')
ax2.set_facecolor('white')
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_color('black')
st.pyplot(fig2)
