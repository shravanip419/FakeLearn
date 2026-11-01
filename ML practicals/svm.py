import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Create dataset
df = pd.DataFrame({
    "StudyHours": [1, 2, 2, 3, 3, 4, 5, 6, 7, 8],
    "Attendance": [40, 45, 50, 50, 55, 65, 70, 75, 85, 90],
    "Result": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
})

print("Dataset:\n", df)

# 2. Split features and target
X = df[["StudyHours", "Attendance"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Train SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# 4. Predict test data
y_pred = model.predict(X_test)

print("\nPredictions:", y_pred)
print("Actual Values:", y_test.values)

# 5. Evaluate model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Fail", "Pass"],
    cmap="Blues"
)

plt.title("SVM Confusion Matrix")
plt.show()

# 7. Predict for user input
hours = float(input("\nEnter Study Hours: "))
attendance = float(input("Enter Attendance (%): "))

new_student = pd.DataFrame({
    "StudyHours": [hours],
    "Attendance": [attendance]
})

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Predicted Result: Pass")
else:
    print("Predicted Result: Fail")
