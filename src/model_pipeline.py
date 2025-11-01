import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from src.data_preprocessing import combinate



def get_pipline():
    cat_category = ["Married", "Self_Employed"]

    # Обробка колонки Gender:
    gender_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("label", OrdinalEncoder())
    ])

    # Обробка колонки Loan_Amount_term:
    term_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler())
    ])

    # Обробка колонок married, self_employed:
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="No")),
        ("encoder", OrdinalEncoder(categories=[["No", "Yes"], ["No", "Yes"]]))
    ])

    # Обробка колонки Credit_History:
    history_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    # Обробка колонки Education:
    edu_transformer = Pipeline(steps=[
        ("encoder", OrdinalEncoder(categories=[["Graduate", "Not Graduate"]]))
    ])

    # Dependents:
    depend_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="0")),
        ("encoder", OrdinalEncoder(categories=[["0", "1", "2", "3"]]))
    ])

    # AppIncome та CoApplicantIncome:
    income_transformer = Pipeline(steps=[
        ("comb", FunctionTransformer(combinate)),
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler())
    ])

    # Loan_Amaount:
    lonamount_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Property_area:
    property_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(drop="first"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("gender", gender_transformer, ["Gender"]),
            ("term", term_transformer, ["Loan_Amount_Term"]),
            ("cat", cat_transformer, cat_category),
            ("history", history_transformer, ["Credit_History"]),
            ("edu", edu_transformer, ["Education"]),
            ("depend", depend_transformer, ["Dependents"]),
            ("income", income_transformer, ["ApplicantIncome", "CoapplicantIncome"]),
            ("loanamount", lonamount_transformer, ["LoanAmount"]),
            ("property", property_transformer, ["Property_Area"])
        ],
        remainder="passthrough"
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(
            class_weight="balanced",
            min_samples_split=5,
            min_samples_leaf=4,
            max_features="sqrt",
            max_depth=10,
            random_state=42)),
    ])

    return pipe