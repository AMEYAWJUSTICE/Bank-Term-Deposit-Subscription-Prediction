# Bank Term Deposit Subscription Prediction

Predict which customers are most likely to subscribe to a term deposit using Machine Learning and deploy the solution with Streamlit.

---

# Project Overview

A Portuguese bank conducted multiple phone marketing campaigns to encourage customers to subscribe to a term deposit product.

Calling every customer is expensive and time-consuming. The marketing team needs a predictive system that identifies customers with the highest likelihood of subscribing so agents can focus on the best leads.

This project builds a complete Machine Learning pipeline to:

* Analyze customer behavior
* Predict subscription likelihood
* Score customers by probability
* Deploy the model as a Streamlit web application

---

# Business Problem

Telemarketing campaigns involve:

* Limited call agents
* Limited campaign time
* High operational costs

The objective is to improve campaign efficiency by targeting customers who are more likely to respond positively.

---

# Project Objectives

## Main Goal

Build a binary classification model that predicts whether a customer will subscribe to a term deposit.

### Target Variable

| Value | Meaning                    |
| ----- | -------------------------- |
| yes   | Customer subscribed        |
| no    | Customer did not subscribe |

---

# Machine Learning Objectives

* Perform customer response prediction
* Reduce unnecessary marketing calls
* Improve campaign conversion rate
* Rank customers based on subscription probability
* Build an interpretable predictive system

---

# Dataset Information

The dataset contains customer demographic, financial, and campaign-related information collected during phone marketing campaigns.

## Common Features

| Feature   | Description                        |
| --------- | ---------------------------------- |
| age       | Customer age                       |
| job       | Type of job                        |
| marital   | Marital status                     |
| education | Education level                    |
| balance   | Account balance                    |
| housing   | Housing loan status                |
| loan      | Personal loan status               |
| contact   | Contact communication type         |
| duration  | Last call duration                 |
| campaign  | Number of contacts during campaign |
| previous  | Previous campaign contacts         |
| poutcome  | Previous campaign outcome          |
| y         | Target variable                    |

---

# Project Pipeline

```text
1. Business Understanding
2. Data Collection
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Feature Engineering
6. Data Preprocessing
7. Handling Class Imbalance
8. Model Training
9. Hyperparameter Tuning
10. Model Evaluation
11. Explainability
12. Deployment with Streamlit
13. Monitoring and Improvement
```

---

# Technologies Used

## Programming Language

* Python

## Libraries

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* plotly
* joblib
* streamlit
* shap

---

# Machine Learning Models

The following classification algorithms can be tested:

| Model                  | Purpose                       |
| ---------------------- | ----------------------------- |
| Logistic Regression    | Baseline model                |
| Decision Tree          | Rule-based prediction         |
| Random Forest          | Ensemble learning             |
| XGBoost                | Gradient boosting             |
| LightGBM               | Fast boosting model           |
| CatBoost               | Categorical feature handling  |
| Support Vector Machine | Margin classification         |
| K-Nearest Neighbors    | Distance-based classification |

---

# Evaluation Metrics

Because the dataset is usually imbalanced, multiple evaluation metrics are important.

| Metric           | Purpose                         |
| ---------------- | ------------------------------- |
| Accuracy         | Overall correctness             |
| Precision        | Quality of positive predictions |
| Recall           | Ability to identify subscribers |
| F1-Score         | Balance of precision and recall |
| ROC-AUC          | Probability ranking quality     |
| Confusion Matrix | Error analysis                  |

---

# Exploratory Data Analysis (EDA)

EDA includes:

* Missing value analysis
* Distribution analysis
* Correlation analysis
* Customer demographic analysis
* Subscription trend analysis
* Campaign effectiveness analysis

### Example Visualizations

* Age distribution
* Job category frequency
* Subscription count plots
* Correlation heatmaps
* Campaign success charts

---

# Feature Engineering

Possible feature engineering tasks:

* Encoding categorical variables
* Scaling numerical variables
* Creating customer segmentation features
* Interaction features
* Campaign efficiency indicators

---

# Handling Class Imbalance

Since subscription responses are often imbalanced:

* SMOTE
* Class weighting
* Undersampling
* Oversampling

may be applied.

---

# Model Training Workflow

```python
# Example Workflow

1. Load dataset
2. Split features and target
3. Train-test split
4. Preprocessing pipeline
5. Train model
6. Evaluate model
7. Save model
```

---

# Saving the Model

```python
import joblib

joblib.dump(model, "model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
```

---

# Streamlit Deployment

## Install Dependencies

```bash
pip install streamlit pandas scikit-learn joblib
```

---

## Run the Application

```bash
streamlit run app.py
```

---

# Example Streamlit Interface

The Streamlit application allows users to:

* Enter customer information
* Predict subscription probability
* Display prediction confidence
* Assist campaign prioritization

---

# Project Structure

```text
bank-term-deposit-prediction/
│
├── app.py
├── model.pkl
├── preprocessor.pkl
├── requirements.txt
├── README.md
│
├── data/
│   └── bank.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── models/
│   └── trained_models/
│
├── visuals/
│   └── charts/
│
└── src/
    ├── preprocessing.py
    ├── training.py
    ├── evaluation.py
    └── prediction.py
```

---

# Installation Guide

## 1. Clone Repository

```bash
git clone <repository-url>
```

---

## 2. Navigate to Project Folder

```bash
cd bank-term-deposit-prediction
```

---

## 3. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4. Install Requirements

```bash
pip install -r requirements.txt
```

---

# Example Prediction Output

| Customer   | Probability | Prediction          |
| ---------- | ----------- | ------------------- |
| Customer A | 87%         | Likely Subscriber   |
| Customer B | 12%         | Unlikely Subscriber |

---

# Business Impact

This solution helps:

* Reduce marketing costs
* Improve campaign efficiency
* Increase subscription conversion rates
* Prioritize high-value leads
* Support data-driven marketing decisions

---

# Future Improvements

Possible future enhancements:

* Real-time API deployment
* Cloud deployment
* AutoML optimization
* Deep learning integration
* Customer segmentation clustering
* Explainable AI dashboards
* Live campaign monitoring

---

# Deployment Options

The project can be deployed using:

* Streamlit Cloud
* Heroku
* Render
* AWS
* Azure
* Docker

---

# Learning Outcomes

This project demonstrates:

* End-to-end machine learning workflow
* Data preprocessing
* Feature engineering
* Classification modeling
* Model evaluation
* Deployment with Streamlit
* Business-oriented AI solutions

---

# Author

Developed as an end-to-end Machine Learning and Data Science project for marketing campaign optimization.

---

# License

This project is for educational and research purposes.
