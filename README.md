# Student Dropout Determinants in Kenyan Higher Education

##  Project Overview
Student dropout in Kenyan higher education institutions is a critical challenge with significant social and economic implications. This project utilizes data science and machine learning to analyze the multifaceted factors—financial, demographic, and academic—that contribute to student attrition. 

By identifying students at high risk of dropping out, educational institutions and policymakers can implement data-driven retention strategies to ensure academic success and efficient resource allocation.

##  Dataset Description
The analysis is performed on a dataset containing over **118,000 student records**. Key features include:
- **Demographics:** Gender, Age, County of Origin.
- **Financials:** Program Cost, Total Loan Allocated, Scholarship Status, Loan-to-Cost Ratio.
- **Academic/Institutional:** Course Category, Exam Year, University Type (Public/Private), Sponsorship (Government/Self).
- **Family Background:** Parental education levels (Mother, Father, and Highest Educational Level).

##  Technology Stack
- **Language:** Python
- **Libraries:** - Data Processing: `pandas`, `numpy`
    - Visualization: `matplotlib`, `seaborn`
    - Statistics: `scipy.stats` (Chi-square, T-tests)
    - Machine Learning: `scikit-learn`, `xgboost`, `imbalanced-learn`
- **Model Storage:** `joblib`, `pickle`

##  Data Pipeline
### 1. Data Cleaning & Preprocessing
- **Handling Missing Values:** Dropped rows with missing critical identifiers or target variables.
- **Column Pruning:** Removed irrelevant identifiers (Bursary batch numbers, indices, etc.).
- **Normalization:** Standardized inconsistent county names and categorical strings.
- **Outlier Management:** Clipped financial ratios and filtered valid student age ranges (17–25).

### 2. Feature Engineering
- **Age Extraction:** Calculated student age from birthdates.
- **Loan-to-Cost Ratio:** Created a feature representing the percentage of program costs covered by loans.
- **Categorical Encoding:** Applied Label Encoding and One-Hot Encoding for model readiness.

### 3. Exploratory Data Analysis (EDA)
- Analyzed dropout distribution by **County** using color-coded bar charts.
- Explored the correlation between **Parental Education** and student funding.
- Visualized **Loan Coverage** across different university types.

### 4. Statistical Hypothesis Testing
- **Chi-Square Tests:** Confirmed significant associations between Scholarship Application/Parental Education and Dropout status.
- **Independent T-Tests:** Validated the impact of Loan Allocation amounts on student retention.

### 5. Machine Learning Modeling
Implemented and compared multiple models, with a focus on handling class imbalance:
- **Logistic Regression** (Baseline)
- **Random Forest** (with class weights)
- **Balanced Random Forest**
- **XGBoost** (Optimized with `scale_pos_weight`)
- **Stacking Ensemble** (Combining base learners for superior performance)

##  Key Findings & Performance
- **Primary Metric:** The project optimizes for **Recall** and **F2-Score** to minimize "False Negatives" (missing a student who is likely to drop out).
- **Top Model:** XGBoost and Stacking Ensembles provided the best trade-off between Precision and Recall.
- **Feature Importance:** Financial burden (Loan-to-Cost ratio) and Parental Education were identified as high-impact predictors.

##  Repository Structure
- `notebook.ipynb`: Main Jupyter Notebook containing the full analysis.
- `best_xgb.pkl` / `model.pkl`: Serialized versions of the trained XGBoost model.
- `cleaned.csv`: The processed dataset used for modeling.

---
**Authors:** 

Natasha Wangari - Team Lead

Elvis Okeyo - Data Scientist

Antony Wala - Data Analyst

Geoffery Kemboi - Scrum Master
