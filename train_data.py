import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report

# Load data
df = pd.read_csv("behavior.csv")

#regression
X_reg = df.drop(["productivity_score", "burnout_level", "date", "day_name"], axis=1)
y_reg = df["productivity_score"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)

pred_r = reg_model.predict(X_test_r)

print("Regression Results")
print("R2 Score:", r2_score(y_test_r, pred_r))
print("MAE:", mean_absolute_error(y_test_r, pred_r))


# ------------------ CLASSIFICATION ------------------ #
le = LabelEncoder()
df["burnout_encoded"] = le.fit_transform(df["burnout_level"])

X_clf = df.drop(
    ["burnout_level", "burnout_encoded", "date", "day_name"], axis=1
)
y_clf = df["burnout_encoded"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_train_c, y_train_c)

pred_c = clf_model.predict(X_test_c)

print("\nClassification Results")
print("Accuracy:", accuracy_score(y_test_c, pred_c))
print(classification_report(y_test_c, pred_c, zero_division=0))


# ------------------ SAVE MODELS ------------------ #
pickle.dump(reg_model, open("productivity_model.pkl", "wb"))
pickle.dump(clf_model, open("burnout_model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("\nModels saved successfully.")