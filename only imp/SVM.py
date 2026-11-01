import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

# 1. Create dataset
data = {
    'CGPA': [5.0, 5.5, 6.0, 6.2, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
    'Aptitude': [40, 45, 50, 52, 55, 65, 70, 75, 85, 90],
    'Placement': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# 2. Separate X and y
X = df[['CGPA', 'Aptitude']]
y = df['Placement']

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Create and train SVM model
model = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear')
)

model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate model
print("\nActual Values:", list(y_test))
print("Predicted Values:", y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 8. Predict for new data
new_data = pd.DataFrame({
    'CGPA': [7.5],
    'Aptitude': [75]
})

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Student is Placed")
else:
    print("Student is Not Placed")

# 9. Visualization - Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM - Confusion Matrix")
plt.show()

# 10. Visualization - Actual Data
plt.figure(figsize=(6, 4))

plt.scatter(
    df['CGPA'],
    df['Aptitude'],
    c=df['Placement'],
    cmap='coolwarm',
    s=100
)

plt.xlabel("CGPA")
plt.ylabel("Aptitude Score")
plt.title("SVM - Student Placement Classification")
plt.colorbar(label="Placement (0 = No, 1 = Yes)")
plt.show()