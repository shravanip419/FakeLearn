
# EXPERIMENT 3:
# Decision Tree (Gini), Random Forest (Bagging)
# and AdaBoost (Boosting)

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 2. Create Dataset (10 Rows)
data = {
    'Age': [22, 25, 28, 30, 35, 40, 45, 50, 55, 60],
    'Income': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    'Purchase': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# 3. Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Separate Features and Target
X = df[['Age', 'Income']]
y = df['Purchase']

# 5. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# 6. Create Three Models

# Decision Tree using Gini Index
dt_model = DecisionTreeClassifier(
    criterion='gini',
    random_state=42
)

# Random Forest (Bagging)
rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    random_state=42
)

# AdaBoost (Boosting)
ada_model = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)

# 7. Store Models
models = {
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'AdaBoost': ada_model
}

# 8. Train, Predict and Evaluate Models
results = {}

predictions = {}

for name, model in models.items():

    # Train Model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print("\n------------------------------")
    print(name)
    print("------------------------------")

    print("Actual Values:", y_test.values)
    print("Predicted Values:", y_pred)

    print("Accuracy:", accuracy)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    # Classification Report
    print("Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=['No Purchase', 'Purchase'],
        zero_division=0
    ))

# 9. Display Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, model) in zip(axes, models.items()):

    cm = confusion_matrix(
        y_test,
        predictions[name]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No', 'Yes'],
        yticklabels=['No', 'Yes'],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 10. Compare Accuracy of All Models
comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': list(results.values())
})

print("\nFINAL MODEL COMPARISON:")
print(comparison)

# 11. Accuracy Comparison Graph
plt.figure(figsize=(8, 5))

plt.bar(
    results.keys(),
    results.values()
)

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.title("Decision Tree vs Random Forest vs AdaBoost")

plt.ylim(0, 1)
plt.grid(axis='y')

plt.show()

# 12. Decision Tree Visualization
plt.figure(figsize=(12, 7))

plot_tree(
    dt_model,
    filled=True,
    rounded=True,
    feature_names=X.columns,
    class_names=['No Purchase', 'Purchase']
)

plt.title("Decision Tree using Gini Index")
plt.show()
