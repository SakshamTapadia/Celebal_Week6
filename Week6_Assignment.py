# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

# Load dataset and preprocess
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train baseline models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

metrics = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": []
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics["Model"].append(name)
    metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
    metrics["Precision"].append(precision_score(y_test, y_pred))
    metrics["Recall"].append(recall_score(y_test, y_pred))
    metrics["F1 Score"].append(f1_score(y_test, y_pred))

# Hyperparameter tuning
# GridSearchCV for SVM
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=5, scoring='f1')
grid_svm.fit(X_train, y_train)

# RandomizedSearchCV for Random Forest
param_dist_rf = {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
rand_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist_rf, n_iter=10, cv=5, scoring='f1', random_state=42)
rand_rf.fit(X_train, y_train)

# Evaluate tuned models
best_svm = grid_svm.best_estimator_
best_rf = rand_rf.best_estimator_

for name, model in [("Tuned SVM", best_svm), ("Tuned RF", best_rf)]:
    y_pred = model.predict(X_test)
    metrics["Model"].append(name)
    metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
    metrics["Precision"].append(precision_score(y_test, y_pred))
    metrics["Recall"].append(recall_score(y_test, y_pred))
    metrics["F1 Score"].append(f1_score(y_test, y_pred))

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics)
metrics_df


# Assuming you already have metrics_df as in the previous step
plt.figure(figsize=(12, 6))
bar_width = 0.2
x = np.arange(len(metrics_df["Model"]))

plt.bar(x - 1.5 * bar_width, metrics_df["Accuracy"], width=bar_width, label='Accuracy')
plt.bar(x - 0.5 * bar_width, metrics_df["Precision"], width=bar_width, label='Precision')
plt.bar(x + 0.5 * bar_width, metrics_df["Recall"], width=bar_width, label='Recall')
plt.bar(x + 1.5 * bar_width, metrics_df["F1 Score"], width=bar_width, label='F1 Score')

plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Comparison of Model Performance Metrics")
plt.xticks(x, metrics_df["Model"], rotation=45)
plt.ylim(0.9, 1.05)
plt.legend()
plt.tight_layout()
plt.show()

