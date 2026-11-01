
# EXPERIMENT 5: SUPPORT VECTOR MACHINE (SVM)
# Predict student placement using CGPA and Aptitude Score

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 2. Create Dataset (10 Rows)
data = {
    'CGPA': [5.0, 5.5, 6.0, 6.2, 6.5,
             7.0, 7.5, 8.0, 8.5, 9.0],

    'Aptitude': [40, 45, 50, 52, 55,
                 65, 70, 75, 85, 90],

    'Placement': [0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# 3. Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Separate Features and Target
X = df[['CGPA', 'Aptitude']]
y = df['Placement']

# 5. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# 6. Feature Scaling and SVM Model
# Pipeline performs scaling before SVM
svm_model = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear')
)

# 7. Train SVM Model
svm_model.fit(X_train, y_train)

# 8. Predict Using SVM
svm_pred = svm_model.predict(X_test)

print("\nSVM Predictions:")
print(svm_pred)

print("Actual Values:")
print(y_test.values)

# 9. Evaluate SVM
svm_accuracy = accuracy_score(y_test, svm_pred)

print("\nSVM Accuracy:", svm_accuracy)

svm_cm = confusion_matrix(y_test, svm_pred)

print("\nSVM Confusion Matrix:")
print(svm_cm)

print("\nSVM Classification Report:")
print(classification_report(
    y_test, svm_pred,
    target_names=['Not Placed', 'Placed'],
    zero_division=0
))

# 10. Plot SVM Confusion Matrix
plt.figure(figsize=(6, 4))

sns.heatmap(
    svm_cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Placed', 'Placed'],
    yticklabels=['Not Placed', 'Placed']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.show()

# 11. Hyperparameter Tuning Using GridSearchCV

pipeline = make_pipeline(
    StandardScaler(),
    SVC()
)

params = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto']
}

grid = GridSearchCV(
    pipeline,
    params,
    cv=2,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

# 12. Display Best Parameters
print("\nBest Parameters:")
print(grid.best_params_)

print("\nBest Cross Validation Score:")
print(grid.best_score_)

# 13. Predict Using Best Model
best_model = grid.best_estimator_

grid_pred = best_model.predict(X_test)

print("\nGridSearchCV Predictions:")
print(grid_pred)

# 14. Evaluate Tuned SVM
grid_accuracy = accuracy_score(y_test, grid_pred)

print("\nSVM with GridSearchCV Accuracy:")
print(grid_accuracy)

grid_cm = confusion_matrix(y_test, grid_pred)

print("\nGridSearchCV Confusion Matrix:")
print(grid_cm)

print("\nGridSearchCV Classification Report:")
print(classification_report(
    y_test, grid_pred,
    target_names=['Not Placed', 'Placed'],
    zero_division=0
))

# 15. Plot GridSearchCV Confusion Matrix
plt.figure(figsize=(6, 4))

sns.heatmap(
    grid_cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Not Placed', 'Placed'],
    yticklabels=['Not Placed', 'Placed']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM with GridSearchCV Confusion Matrix")
plt.show()

# 16. Compare Both Models
comparison = {
    'SVM': svm_accuracy,
    'SVM with GridSearchCV': grid_accuracy
}

print("\nFinal Model Comparison:")
for name, score in comparison.items():
    print(name, ":", score)

# 17. Comparison Graph
plt.figure(figsize=(8, 5))

plt.bar(
    comparison.keys(),
    comparison.values()
)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("SVM vs SVM with GridSearchCV")

plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()
