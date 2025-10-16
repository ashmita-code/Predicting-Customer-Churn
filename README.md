# 📊 Customer Churn Prediction using Machine Learning

### 🧠 Project Overview

This project predicts customer churn using multiple machine learning
models and data balancing techniques.\
It includes exploratory data analysis (EDA), feature correlation, model
comparison, threshold optimization, and performance evaluation using
precision--recall and ROC curves.

------------------------------------------------------------------------

## 🏗️ Project Pipeline

### **1. Data Understanding & Preparation**

-   Loaded and cleaned the Telco Customer Churn dataset.\
-   Checked missing values and data consistency.\
-   Created helper columns like `tenure_group` and `tenure_year_group`
    for categorical bucketing.\
-   Encoded categorical variables using **One-Hot Encoding** for model
    readiness.

------------------------------------------------------------------------

### **2. Exploratory Data Analysis (EDA)**

-   Performed detailed univariate and bivariate visualizations:
    -   **Histograms & boxplots** for numerical distributions.
    -   **Bar charts & pie charts** for categorical variables.
    -   **Heatmap of correlations** to identify highly correlated
        features.
-   Key Observations:
    -   `TotalCharges`, `tenure`, and `MonthlyCharges` had the highest
        influence on churn.
    -   Customers on month-to-month contracts and electronic check
        payments showed higher churn rates.

------------------------------------------------------------------------

### **3. Feature Engineering**

-   Applied **one-hot encoding** to multi-category columns (`Contract`,
    `PaymentMethod`, etc.).
-   Ensured all features were numeric for model training.
-   Verified dataset split and shape:
    -   `Train shape: (5625, 30)`\
    -   `Test shape: (1407, 30)`

------------------------------------------------------------------------

### **4. Model Building & Evaluation**

#### **Model 1: Logistic Regression**

-   Achieved **Accuracy: 0.804**
-   Precision: 0.65 \| Recall: 0.57 \| F1: 0.61\
-   Identified that logistic regression performs well but slightly
    underfits on minority churn class.

#### **Model 2: Random Forest**

-   Achieved **Accuracy: 0.79**
-   Precision: 0.63 \| Recall: 0.50 \| F1: 0.56\
-   Provided insights into **feature importance**, where `TotalCharges`,
    `tenure`, and `MonthlyCharges` dominated.

#### **Model 3: XGBoost (after SMOTE balancing)**

-   Balanced dataset using **SMOTE**: equalized churn vs. non-churn
    samples (4130 each).\
-   XGBoost delivered strong and stable results:
    -   **Accuracy: 0.787 (default threshold)**\
    -   **Recall: 0.65 \| Precision: 0.59 \| F1: 0.62**\
-   Outperformed previous models in identifying true churners.

------------------------------------------------------------------------

### **5. Threshold Optimization**

-   Tested multiple probability thresholds (0.5 → 0.4 → optimized
    0.447).\
-   Found **best threshold = 0.447** using F1-maximization.
-   Comparison: \| Metric \| Default (0.5) \| Optimized (0.447) \|
    \|---------\|----------------\|-------------------\| \| Accuracy \|
    0.787 \| 0.777 \| \| Recall (Churn=1) \| 0.65 \| 0.69 \| \|
    Precision (Churn=1) \| 0.59 \| 0.57 \|

📈 **Interpretation:**\
Lowering the threshold increases recall (more churners caught) at a
small precision trade-off --- ideal for proactive retention strategies.

------------------------------------------------------------------------

### **6. Visual Evaluation**

-   **Precision--Recall vs Threshold Curve:** to visualize the
    precision--recall trade-off.\
-   **Bar Comparison Plot:** compared metrics between thresholds (0.5
    vs. 0.447).\
-   **ROC and Precision--Recall Curves:**
    -   ROC-AUC = **0.83** → Excellent class separation.\
    -   AUC-PR = **0.636** → Solid performance on imbalanced data.

------------------------------------------------------------------------

### **7. Key Findings**

-   Customers with short tenure, high monthly charges, and electronic
    payment modes are more likely to churn.\
-   The optimized XGBoost model with a **threshold of 0.447** offers the
    best recall for churn detection.\
-   ROC and PR curves confirm strong classifier reliability.

------------------------------------------------------------------------

## 🧩 Tools & Libraries

-   **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**\
-   **Scikit-learn**, **Imbalanced-learn (SMOTE)**, **XGBoost**

------------------------------------------------------------------------

## 🏁 Final Summary

✅ Conducted extensive EDA and correlation analysis.\
✅ Compared Logistic Regression, Random Forest, and XGBoost models.\
✅ Balanced classes with SMOTE for fair training.\
✅ Tuned probability threshold for optimal recall.\
✅ Validated model with AUC-ROC and AUC-PR metrics.

> **Outcome:** XGBoost (threshold = 0.447) achieved the best
> recall--precision balance --- allowing the business to catch at-risk
> customers earlier and improve retention campaigns.
