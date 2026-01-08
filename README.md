# Telco-Customer-Churn-Machine-Learning-Project

## 📌 Project Overview
This project focuses on building an **end-to-end machine learning pipeline** to predict customer churn using the **Telco Customer Churn dataset**.  
Customer churn prediction is a critical business problem where the goal is to identify customers who are likely to leave a service.

The project emphasizes **real-world data preprocessing**, **handling class imbalance**, and **model evaluation beyond accuracy**.

---

## 📊 Dataset
- **Source:** Kaggle – Telco Customer Churn Dataset  [Kaggle Data Link]https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Rows:** ~7,000 customers  
- **Target Variable:** `Churn` (Yes / No)

Each row represents a customer with demographic details, service usage, and billing information.

---

## 🧠 Problem Statement
To build a classification model that predicts whether a customer will churn based on their service and account information.

---

## 🛠️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## 🔄 Project Workflow

### 1️⃣ Data Cleaning
- Converted `TotalCharges` from string to numeric
- Handled missing values
- Removed hidden whitespace issues in categorical columns
- Dropped non-informative identifier column (`customerID`)

---

### 2️⃣ Feature Engineering
- One-hot encoded categorical variables using `pd.get_dummies`
- Avoided dummy variable trap using `drop_first=True`
- Converted target variable (`Churn`) to binary values (0 / 1)

---

### 3️⃣ Train-Test Split
- Used an 80–20 split
- Applied **stratified sampling** to preserve class distribution

---

### 4️⃣ Models Implemented
The following models were trained and compared:

- Logistic Regression (baseline model)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest

Class imbalance was handled using `class_weight='balanced'` where applicable.

---

### 5️⃣ Model Evaluation
Instead of relying only on accuracy, the models were evaluated using:
- Precision
- Recall (especially for churn class)
- F1-score
- ROC-AUC

This approach ensures meaningful performance evaluation for imbalanced datasets.

---
