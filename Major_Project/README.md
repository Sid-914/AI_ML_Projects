## Introduction:

This project is focused on predicting real estate prices using a variety of machine learning algorithms. The dataset comprises 60,000 housing records across Indian cities, with features like property size, number of bedrooms and bathrooms, furnishing type, and more. The objective is to build a predictive model that can estimate housing prices accurately, enabling insights for buyers, sellers, and developers.
Project Description The system uses regression models to predict house prices based on structured data. It involves:
Cleaning and preprocessing real estate data
Visualizing patterns in pricing and features
Training multiple ML models like Linear Regression, Decision Trees, Random Forest, XGBoost, LightGBM, etc.
Evaluating performance using metrics like RMSE, MAE, and R²

# 🏡 Real Estate Price Prediction Using Machine Learning

This project focuses on predicting house prices using machine learning algorithms applied to a structured real estate dataset. It explores how various features like location, property size, furnishing type, number of bedrooms and bathrooms, and more influence the final price of a property.

---

## 📁 Dataset

- **Rows**: 60,000 property listings
- **Columns**: 11 features including:
  - `Size (sq ft)`
  - `Bedrooms`
  - `Bathrooms`
  - `Furnishing`
  - `Age of House (years)`
  - `Location`
  - `Parking Available`
  - `Type of Property`
  - `Nearby Facilities`
  - `Price` (target)

---

## 🚀 Features

- Clean and preprocess raw real estate data
- Encode categorical variables using `LabelEncoder`
- Feature scaling using `StandardScaler`
- Data visualization using Seaborn and Matplotlib
- Correlation analysis through heatmaps
- Comparison of multiple machine learning regression models
- Evaluation using RMSE, MAE, MSE, and R² Score

---

## 🤖 Models Implemented

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- Support Vector Regressor (SVR)  
- K-Nearest Neighbors (KNN)  
- XGBoost  
- LightGBM

---

## 📊 Results Summary

Each model was evaluated using:
- ✅ Root Mean Squared Error (RMSE)
- ✅ Mean Absolute Error (MAE)
- ✅ Mean Squared Error (MSE)
- ✅ R² Score

**Best Performing Models**:
- **XGBoost** and **Gradient Boosting** achieved an R² Score of **0.99**, indicating excellent predictive power.

---

## 📈 Visualizations

The project includes the following visual insights:
- Histogram of house prices
- Boxplot: Bedrooms vs Price
- Scatterplot: Size vs Price
- Violin plot: Bathrooms vs Price
- Swarmplot: Furnishing vs Price
- Lineplot: Age vs Average Price
- Heatmap: Correlation Matrix
- Countplot & Barplot: Type of Property

---

## 🧠 Future Improvements

- Include geolocation (latitude, longitude) for more precise analysis
- Add temporal data (listing date, month, year)
- Try deep learning models (e.g., TensorFlow, Keras)
- Deploy the model via a Flask or Streamlit web app

---

## 🛠️ How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the main Python script:
    ```bash
    python main.py
    ```

> Make sure the file `real_estate_data_final.csv` is present in the same directory.

---

## Conclusion:

This project successfully demonstrates the power and applicability of machine learning algorithms in the domain of real estate price prediction. By leveraging a structured dataset with 60,000 property listings, we implemented a robust data pipeline that includes preprocessing, feature engineering, visualization, and model evaluation. The performance of advanced ensemble methods like XGBoost and Gradient Boosting stood out, delivering R² scores as high as 0.99, indicating exceptional accuracy in predicting housing prices.
Key contributions to this performance include:
Comprehensive preprocessing: Handling categorical data using label encoding and standardizing features ensured models could learn effectively.
Insightful visualizations: Plots like histograms, scatterplots, boxplots, and heatmaps provided critical understanding of feature distributions and relationships.
Thorough model evaluation: Multiple regression techniques were compared using well-established error metrics (RMSE, MAE, MSE, R²).
The workflow and modeling approach presented here is generalizable and can be adapted for other price prediction tasks such as car pricing, rental forecasting, or even insurance premium estimation. By fine-tuning hyperparameters and incorporating advanced techniques such as feature selection or automated machine learning (AutoML), performance could be further enhanced.

