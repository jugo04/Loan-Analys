from src.data_preprocessing import data_preprocessing
from sklearn.model_selection import train_test_split
from src.model_pipeline import get_pipline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib

df = data_preprocessing("../DataFrame/train.csv")

X = df.drop("Loan_Status", axis =1)
y = df["Loan_Status"].map({"N": 0, "Y": 1})

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

pipe = get_pipline()

pipe.fit(X_train, y_train)

#Поріг погодження кредиту при вірогідності 35%:
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

joblib.dump(pipe, "../model/loan_model.pkl")
print("Model saved")