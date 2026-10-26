# 📊 Telecom Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

This project predicts whether a telecom customer is at risk of leaving the company using Machine Learning.

The project includes exploratory data analysis, data preprocessing, model training, model evaluation, threshold optimization, and an interactive Streamlit application for predicting customer churn.

## 🎯 Project Objectives

- Analyze customer churn patterns.
- Identify factors associated with customer churn.
- Build and compare multiple Machine Learning models.
- Improve churn detection through classification threshold optimization.
- Develop an interactive application for predicting customer churn risk.

## 📂 Dataset

**Dataset:** IBM Telco Customer Churn

- Total customers: 7,043
- Original dataset columns: 21
- Customers who churned: 1,869
- Customers who stayed: 5,174
- Overall churn rate: 26.54%

The dataset includes customer demographics, subscription details, contract types, payment methods, and billing information.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## 🔍 Exploratory Data Analysis

The analysis identified several customer segments associated with higher churn rates.

| Customer Segment | Observed Churn Rate |
|---|---:|
| Customers with 0–12 months tenure | 47.4% |
| Month-to-month contracts | 42.7% |
| Fiber optic internet service | 41.9% |
| Electronic check payment method | 45.3% |

These findings highlight customer segments that may benefit from retention analysis.

## 🤖 Machine Learning Models

Three classification algorithms were trained and evaluated.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.1% | 64.8% | 55.1% | 59.5% | 84.1% |
| Decision Tree | 79.6% | 64.1% | 52.4% | 57.6% | 83.0% |
| Random Forest | 78.0% | 60.4% | 49.7% | 54.5% | 81.6% |

**Selected Model:** Logistic Regression

Logistic Regression achieved the highest ROC-AUC score among the three evaluated models.

## ⚙️ Classification Threshold Optimization

The initial model used a classification threshold of 50%.

To improve the identification of customers likely to churn, different thresholds were evaluated using cross-validation on the training data.

The selected threshold was **30%**.

### Optimized Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 74.7% |
| Precision | 51.6% |
| Recall | 74.3% |
| F1-Score | 60.9% |
| ROC-AUC | 84.1% |

Reducing the threshold improved churn recall from approximately **55.1% to 74.3%** on the test set.

This allowed the model to identify more customers who actually churned, although it also increased false-positive predictions.

## 💡 Key Insights

- Customers with shorter tenure showed higher observed churn rates.
- Month-to-month contracts were associated with higher churn.
- Fiber optic customers showed relatively high churn rates.
- Logistic Regression performed best among the evaluated models.
- Threshold optimization improved the detection of customers at risk of churning.

These findings describe associations and model predictions, not proven causes of churn.

## 🖥️ Streamlit Prediction Application

An interactive Streamlit application was developed to predict customer churn risk.

### Application Features

- Accepts customer information through an interactive form.
- Loads a previously trained Machine Learning model.
- Predicts customer churn probability.
- Classifies customers as **At Risk** or **Not Flagged**.
- Uses an optimized 30% risk classification threshold.

The application allows users to test different customer profiles without retraining the model.

## ▶️ How to Run the Application

**1. Clone or download this repository.**

**2. Install the required libraries:**

```bash
pip install -r requirements.txt
```

**3. Start the Streamlit application:**

```bash
streamlit run telecom_churn_app.py
```

**4. Open the local address displayed in your terminal.**

Enter customer details and click **Predict Customer Churn** to view the prediction.

## 📁 Project Files

| File | Description |
|---|---|
| `TelecomPythonFile.py` | Data analysis, preprocessing, model training, and evaluation |
| `telecom_churn_app.py` | Interactive Streamlit application |
| `telecom_churn_model.pkl` | Saved Logistic Regression model |
| `churn_threshold.pkl` | Selected classification threshold |
| `model_features.pkl` | Required input feature names |
| `requirements.txt` | Python dependencies |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Telecom churn dataset |

## ⚠️ Limitations

- The model was evaluated using a historical telecom churn dataset.
- Predictions may not generalize to every telecom company.
- Churn probabilities are estimates, not guaranteed outcomes.
- The application is intended as a Machine Learning demonstration.
- Model performance should be monitored and validated before real-world business deployment.

## 📌 Conclusion

This project demonstrates an end-to-end Machine Learning workflow, from exploratory data analysis and model comparison to threshold optimization and application development.

The final Logistic Regression model achieved **84.1% ROC-AUC**, while threshold optimization increased churn recall to **74.3%**.

The Streamlit application demonstrates how the trained model can be used to generate customer-level churn risk predictions.
