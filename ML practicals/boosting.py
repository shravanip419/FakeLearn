
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Create DataFrame
data = {
    "StudyHours": [1, 2, 2, 3, 3, 4, 5, 6, 7, 8],
    "Attendance": [40, 45, 50, 50, 55, 65, 70, 75, 85, 90],
    "Result": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# Separate features and target
X = df[["StudyHours", "Attendance"]]
y = df["Result"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Define models
models = {
    "Decision Tree (Gini)": DecisionTreeClassifier(
        criterion="gini", random_state=42
    ),
    "AdaBoost (Boosting)": AdaBoostClassifier(
        n_estimators=50,
        learning_rate=0.5,
        random_state=42
    )
}

accuracies = {}

# Train, predict and display individual confusion matrices
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    accuracies[name] = acc

    print("\n", name)
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"]
    )
    plt.title(name + " - Confusion Matrix")
    plt.xlabel("Predicted Result")
    plt.ylabel("Actual Result")
    plt.show()

# Compare accuracy
print("\nAccuracy Comparison:")
for name, acc in accuracies.items():
    print(name, ":", round(acc * 100, 2), "%")
