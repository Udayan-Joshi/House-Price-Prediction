# 🏡 House Price Prediction using XGBoost


Predicting housing prices in California using powerful regression techniques and real-world datasets.

---

Try the live application : https://house-price-prediction-0205.streamlit.app/

---

## 📌 Project Description

This project builds a regression model using the **XGBoost Regressor** to predict **house prices** in California. The model is trained on the [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset), with steps including data analysis, visualization, training, and evaluation.

---

## 🚀 Features

- 📥 Fetches and processes the California Housing dataset
- 📊 Visualizes data with correlation heatmaps and scatter plots
- 🧠 Trains an XGBoost Regression model
- 📈 Evaluates performance using R² Score and MAE
- 🖼️ Displays actual vs predicted values

---

## 📂 Dataset Information

| Feature        | Description                            |
|----------------|----------------------------------------|
| `MedInc`       | Median income in the block group       |
| `HouseAge`     | Median house age in the block group    |
| `AveRooms`     | Average number of rooms per household  |
| `AveBedrms`    | Average number of bedrooms             |
| `Population`   | Total population in the block group    |
| `AveOccup`     | Average number of occupants per house  |
| `Latitude`     | Latitude coordinate                    |
| `Longitude`    | Longitude coordinate                   |
| `Price`        | Median house value *(Target)*          |

---

## 📊 Model Performance

| Metric                  | Training Set | Test Set  |
|-------------------------|--------------|-----------|
| R² Score                | 0.94         | 0.83      |
| Mean Absolute Error     | 0.19         | 0.31      |

✅ **High accuracy** on both training and test sets, indicating a strong model without significant overfitting.

---

## 🧪 Tech Stack

- **Language:** Python 3.8+
- **Libraries:** 
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
  - `xgboost`

---

## 📷 Sample Output

![Scatter Plot](https://github.com/user-attachments/assets/b642fdf9-69f9-48b1-aedd-4fc043b5717e)


---


