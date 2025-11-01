import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


#Завантажуємо датасет:
df = pd.read_csv("DataFrame/train.csv")

#Видалено колонку яка не впливає на аналіз:
df = df.drop("Loan_ID", axis =1)
df["Dependents"] = df["Dependents"].replace("3+", "3")

X = df.drop("Loan_Status", axis =1)
y = df["Loan_Status"].map({"N": 0, "Y": 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_category = ["Married", "Self_Employed"]
income = ["ApplicantIncome", "CoapplicantIncome"]

def combinate(x):
    x = x.copy()
    x["TotalIncome"] = x["ApplicantIncome"] + x["CoapplicantIncome"]
    return x[["TotalIncome"]]

#Обробка колонки Gender:
gender_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("label", OrdinalEncoder())
])

#Обробка колонки Loan_Amount_term:
term_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", StandardScaler())
])

#Обробка колонок married, self_employed:
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="No")),
    ("encoder", OrdinalEncoder(categories=[["No","Yes"], ["No","Yes"]]))
])

#Обробка колонки Credit_History:
history_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

#Обробка колонки Education:
edu_transformer = Pipeline(steps=[
    ("encoder", OrdinalEncoder(categories=[["Graduate", "Not Graduate"]]))
])

#Dependents:
depend_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="0")),
    ("encoder", OrdinalEncoder(categories=[["0","1","2","3"]]))
])

#AppINcome та CoapplicantIncome:
income_transformer = Pipeline(steps=[
    ("comb", FunctionTransformer(combinate)),
    ("log", FunctionTransformer(np.log1p)),
    ("scaler", StandardScaler())
])

#Loan_Amaount:
lonamount_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean", missing_values=np.nan)),
    ("scaler", StandardScaler())
])

#Propery_area:
property_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(drop="first"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("tender", gender_transformer, ["Gender"]),
        ("term", term_transformer, ["Loan_Amount_Term"]),
        ("cat", cat_transformer, cat_category),
        ("history", history_transformer, ["Credit_History"]),
        ("edu", edu_transformer, ["Education"]),
        ("depend", depend_transformer, ["Dependents"]),
        ("income", income_transformer, income),
        ("loanamount", lonamount_transformer, ["LoanAmount"]),
        ("property", property_transformer, ["Property_Area"])
    ],
    remainder="passthrough"
)

pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(
        class_weight= "balanced",
        min_samples_split = 5,
        min_samples_leaf = 4,
        max_features = "sqrt",
        max_depth = 10,
        random_state=42)),
])

pipe.fit(X_train, y_train)

y_proba_class0 = pipe.predict_proba(X_test)[:, 0]
best_threshold = 0.35
y_predict = (y_proba_class0 >= best_threshold).astype(int)
y_predict = 1 - y_predict

print("Accuracy:", pipe.score(X_test, y_test))
print(classification_report(y_test, y_predict))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predict))

cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')

print("=== CROSS-VALIDATION ===")
print(f"CV scores по кожному фолду: {cv_scores}")
print(f"Середня CV accuracy: {cv_scores.mean():.3f}")
print(f"Стандартне відхилення: {cv_scores.std():.3f}")