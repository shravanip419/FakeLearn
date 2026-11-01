import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Create Dataset
data = {
    'Age': [22, 25, 28, 30, 35, 40, 45, 50, 55, 60],
    'Income': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    'Purchase': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

# 2. Separate Features and Target
X = df[['Age', 'Income']]
y = df['Purchase']

# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Create AdaBoost Model
model = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)

# 5. Train Model
model.fit(X_train, y_train)

# 6. Prediction
y_pred = model.predict(X_test)

print("Predicted:", y_pred)
print("Actual:", y_test.values)

# 7. Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 8. Visualization
plt.bar(['AdaBoost'], [accuracy])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Boosting - AdaBoost")
plt.show()