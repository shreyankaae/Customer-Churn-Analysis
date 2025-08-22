# Customer Churn Intelligence: ML-Powered Dashboard

## Project Overview

This project uses the **Telco Customer Churn (IBM) dataset** to build a machine learning solution that predicts whether a customer will churn. The goal is to help businesses (telecom, SaaS, banking, retail) **retain high-value customers** by understanding **who is at risk, why they churn, and what can be done**.

Key project deliverables:

* **Exploratory Data Analysis (EDA)** – uncover trends and churn drivers.
* **Machine Learning Models** – Logistic Regression, Random Forest, XGBoost.
* **Model Explainability** – SHAP values to interpret churn predictions.
* **Dashboard Visualization** – insights presented via Tableau and Streamlit.
* **Deployment** – interactive web app on Streamlit Cloud.

---

## Dataset

**Telco Customer Churn (IBM sample dataset)**

* **Rows:** \~7043 customers
* **Columns:** Demographics, services, contract details, billing, churn status
* **Target Variable:** `Churn` (Yes / No)

Example features:
* `gender`, `senior_citizen`, `partner`, `dependents`
* `tenure_months`, `monthly_charges`, `total_charges`
* `internet_service`, `contract`, `payment_method`
* `churn_reason` (why customers left)

---

## Tech Stack

| Stage               | Tools & Libraries                          |
| ------------------- | ------------------------------------------ |
| Data Cleaning & EDA | Python, Pandas, Numpy, Seaborn, Matplotlib |
| Modeling            | Scikit-learn, XGBoost, LightGBM            |
| Interpretability    | SHAP                                       |
| Dashboarding        | Tableau Public, Streamlit                  |
| Deployment          | Streamlit Cloud, GitHub                    |

---

## Project Workflow

1. **Data Cleaning**

   * Handle missing values
   * Encode categorical variables (One-Hot Encoding)
   * Scale numerical features

2. **Exploratory Data Analysis (EDA)**

   * Churn distribution
   * Churn vs Contract, Tenure, Monthly Charges
   * Correlation analysis

3. **Modeling**

   * Train/test split (80/20)
   * Logistic Regression baseline
   * Random Forest & XGBoost for performance
   * Hyperparameter tuning with GridSearchCV

4. **Model Evaluation**

   * Accuracy, Precision, Recall, F1, ROC-AUC
   * Confusion Matrix

5. **Explainability**

   * SHAP plots to explain churn drivers (e.g., high monthly charges, contract type)

6. **Dashboarding**

   * Tableau: KPI view (Churn %, Churn by Contract, Tenure distribution)
   * Streamlit: Interactive churn predictor + EDA visualizations

7. **Deployment**

   * Streamlit Cloud hosting
   * GitHub repo for code + documentation

---

## Dashboard KPIs

* **Overall Churn Rate**
* **Churn by Contract Type**
* **Churn by Payment Method**
* **Churn by Tenure Groups**
* **Feature Importance (ML-based)**
* **Customer Segmentation (High vs Low Risk)**

---

##  Learning Resources

1. https://www.ibm.com/think/topics/customer-churn

2. https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

3. https://www.superoffice.com/blog/reduce-customer-churn/

4. https://shap.readthedocs.io/en/latest/

5. https://openml.github.io/automlbenchmark/frameworks.html

---

