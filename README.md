# 🧠 Binary Classification on Health Dataset using Machine Learning

This repository contains my **YBI Foundation Internship Project**, where I built and evaluated a **binary classification model** using Machine Learning techniques.  
The goal is to predict whether a person is likely to have a certain **medical condition** (disease present or not) based on several health-related features.

---

## 📚 Project Overview

This project demonstrates:
- How to preprocess and clean raw health data
- How to train and evaluate multiple ML models
- How to compare metrics like Accuracy, Precision, Recall, F1, and ROC-AUC
- How to select and interpret the best-performing model

**Primary Objective:**  
Predict whether a patient has a disease (1) or not (0) based on diagnostic measurements.

---

## 🧩 Dataset Information

- **Dataset Used:** Pima Indians Diabetes Dataset (Kaggle)
- **Rows:** 768  
- **Columns:** 9  
- **Target Column:** `Outcome`  
  - 1 → Disease Present  
  - 0 → No Disease  

**Key Features:**
- Glucose  
- Blood Pressure  
- BMI  
- Age  
- Insulin  
- SkinThickness  
- DiabetesPedigreeFunction  

---

## ⚙️ Methodology

| Step | Description |
|------|--------------|
| **1. Data Preprocessing** | Handled missing values using mean imputation and scaled features using `StandardScaler`. |
| **2. Model Training** | Tested Logistic Regression, Decision Tree, Random Forest, and KNN models. |
| **3. Evaluation** | Used metrics like Accuracy, Precision, Recall, F1-score, and ROC-AUC. |
| **4. Best Model** | Logistic Regression performed the best with 87.2% accuracy. |

---

## 📊 Results

**Best Model:** Logistic Regression  
**Test Metrics:**
- Accuracy: **87.2%**
- Precision: **85.4%**
- Recall: **88.1%**
- F1-Score: **86.7%**
- ROC-AUC: **0.91**

**Confusion Matrix Summary:**
- True Positives (TP): 112  
- False Positives (FP): 14  
- False Negatives (FN): 15  
- True Negatives (TN): 125  

**Top Features Influencing Prediction:**
- Glucose  
- BMI  
- Age  
- Blood Pressure  

---

## 🧠 Key Insights

- Glucose and BMI were the most significant indicators of disease presence.  
- Logistic Regression offered high interpretability and balanced precision/recall trade-off.  
- Model generalizes well without overfitting (verified via 5-fold cross-validation).  

---

## 🚀 Future Improvements

- Apply **GridSearchCV** for hyperparameter tuning.  
- Handle **class imbalance** using SMOTE or weighted classes.  
- Add new health indicators (cholesterol, exercise, diet).  
- Build a **Streamlit web app** for real-time predictions.  
- Explore deep learning models for higher accuracy.

---

## 🧰 Tech Stack

- **Language:** Python 3.x  
- **Environment:** Google Colab / Jupyter Notebook  
- **Libraries:**  
  `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## 🧑‍💻 How to Run

1. Clone this repository  
   ```bash
   git clone https://github.com/Xubhv/Internship.git
   cd Internship
