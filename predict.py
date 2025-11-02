import joblib
import pandas as pd
from src.data_preprocessing import data_preprocessing

pipline = joblib.load("model/loan_model.pkl")

data, loan_id = data_preprocessing("DataFrame/test.csv")

prediction = pipline.predict(data)

prognosis = pd.DataFrame({
    "Loan_ID": loan_id,
    "Approved": prediction
})

prognosis.to_csv("prognosis.csv", index=False)