import pandas as pd

def combinate(x):
    x = x.copy()
    x["TotalIncome"] = x["ApplicantIncome"] + x["CoapplicantIncome"]
    return x[["TotalIncome"]]

def data_preprocessing(path):
    df = pd.read_csv(path)

    loan_id = df["Loan_ID"]
    df = df.drop("Loan_ID", axis=1)
    df["Dependents"] = df["Dependents"].replace("3+", "3")

    return df, loan_id