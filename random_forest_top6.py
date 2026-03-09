
# ==========================================
# Random Forest - Top 6 Feature Selection
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------
# Replace with your file path
df = pd.read_csv("processed_breast_cancer_data.csv")

# Drop ID column
if "id" in df.columns:
    df = df.drop(columns=["id"])

# The 'diagnosis' column is already numeric (0 and 1), so no mapping is needed here.
# Drop rows where 'diagnosis' is NaN, if any exist
df.dropna(subset=["diagnosis"], inplace=True)

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ------------------------------------------
# 2. Train-Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------
# 3. Train Random Forest on All Features
# ------------------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# ------------------------------------------
# 4. Rank Features
# ------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns)
feature_ranking = importances.sort_values(ascending=False)

print("
Feature Ranking:
")
print(feature_ranking)

# ------------------------------------------
# 5. Select Top 6 Features
# ------------------------------------------
top6_features = feature_ranking.head(6).index
print("
Top 6 Features:
", list(top6_features))

# Keep only Top 6
X_train_top6 = X_train[top6_features]
X_test_top6 = X_test[top6_features]

# ------------------------------------------
# 6. Retrain Model Using Top 6 Features
# ------------------------------------------
rf_top6 = RandomForestClassifier(n_estimators=200, random_state=42)
rf_top6.fit(X_train_top6, y_train)

y_pred = rf_top6.predict(X_test_top6)

# ------------------------------------------
# 7. Evaluation
# ------------------------------------------
print("
Accuracy with Top 6 Features:",
      accuracy_score(y_test, y_pred))

print("
Classification Report:
",
      classification_report(y_test, y_pred))

print("
Confusion Matrix:
",
      confusion_matrix(y_test, y_pred))

# ------------------------------------------
# 8. Plot Top 6 Feature Importances
# ------------------------------------------
plt.figure(figsize=(8, 6))
feature_ranking.head(6).sort_values().plot(kind="barh")
plt.title("Top 6 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 9. Save Reduced Dataset (Optional)
# ------------------------------------------
df_reduced = pd.concat([X[top6_features], y], axis=1)
df_reduced.to_csv("reduced_top6_dataset.csv", index=False)

print("
Reduced dataset saved as 'reduced_top6_dataset.csv'")
