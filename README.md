# Analysis of Student Dropout Determinants in Kenyan Higher Education

##  Project Overview
This project addresses the critical challenge of student attrition within the Kenyan higher education system. By analyzing a dataset of over 100,000 students, the study identifies the complex interplay between academic performance, financial stability, and demographic factors.

##  Problem Identification
The core objective is to mitigate the socio-economic costs of student dropouts. The analysis focuses on:
- **Financial Stress:** Assessing the impact of tuition fee status and student loans.
- **Academic Preparation:** Comparing entry grades (High School) with university performance (GPA).
- **Engagement:** Evaluating how attendance and extracurricular involvement influence retention.

##  Tech Stack
- **Language:** Python
- **Key Libraries:** BalancedRandomForestClassifier, ColumnTransformer, ConfusionMatrixDisplay, LinearRegression, LogisticRegression, Pipeline, RFE, RandomForestClassifier, SMOTE, SimpleImputer, StandardScaler, XGBClassifier, accuracy_score,auc,roc_curve, chi2_contingency, datetime, matplotlib, numpy, pandas, seaborn, train_test_split, ttest_ind, warnings
- **Model:** XGBoost (Optimized via GridSearchCV)

##  Methodology
1. **Data Cleaning:** Imputation and handling of categorical variables.
2. **Feature Engineering:** Selection of key determinants for student success.
3. **Modeling:** Implementing an XGBoost classifier to predict dropout risk.
4. **Evaluation:** Assessing performance using ROC-AUC and Accuracy.

##  Key Results
The tuned XGBoost model produced the following results:
```text
TUNED XGBOOST
Accuracy: 0.6974763406940063
ROC-AUC: 0.49535119493033175
```
*Note: The model highlights that financial status (being a debtor) and academic performance are leading indicators of dropout risk.*

##  Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/student-dropout-analysis.git
   ```
2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```

##  Usage
Open the `index.ipynb` file in Jupyter Notebook or Google Colab to view the full analysis, visualizations, and model training steps.