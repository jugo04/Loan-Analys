# Loan Approval Prediction Project

## Description
This is an educational project aimed at predicting data using machine learning to determine the approval of financial loans.

## About the Dataset
The dataset includes gender, marital status, education, number of dependents, income, loan amount, credit history, and other features. We receive this data from the client when they submit a loan application.  
**Kaggle:** https://www.kaggle.com/datasets/krishnaraj30/finance-loan-approval-prediction-data

## Work Performed
- **Exploratory Data Analysis (EDA)** — investigated missing values, correlations, and data distributions
- **Data Preprocessing**: cleaning, imputation of missing values, encoding of categorical variables, scaling of numerical features
- Created a new feature **TotalIncome** to improve the prediction quality of the model
- Built an **ML pipeline** using ColumnTransformer and Pipeline
- Trained a **DecisionTreeClassifier** model with hyperparameter tuning and testing of different options
- Evaluated the model using metrics: accuracy, precision, recall, f1-score, and built a confusion matrix
- Researched the impact of **threshold tuning** and **feature engineering** on prediction quality

## Project Structure
├── DataFrame/ — input data (train/test CSV files)
├── model/ — saved models
│ └── loan_model.pkl
├── src/ — code for preprocessing and model building
│ ├── data_preprocessing.py
│ ├── pipeline.py
│ └── model_train.py
├── EDA.py — exploratory data analysis
├── predict.py — makes predictions on new data
└── prognosis.csv — prediction results

## Results
- The model demonstrated an average accuracy of approximately **77%** on the test data after hyperparameter optimization and sample balancing
- **Recall for Class 0 = 0.74** - detects 74% of risky clients
- **Balance between classes** - does not skew towards either class
- **Stability** - low standard deviation (0.041)
- The **Confusion Matrix** shows a healthy balance