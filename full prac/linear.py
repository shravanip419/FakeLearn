
# EXPERIMENT 1: LINEAR REGRESSION
# Predict student marks based on study hours

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 2. Create Dataset (10 Rows)
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks': [35, 40, 45, 50, 55, 60, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# 3. Separate Independent and Dependent Variables
X = df['Hours']
y = df['Marks']

# 4. Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 5. Apply Linear Regression using linregress()
slope, intercept, r, p, se = linregress(X_train, y_train)

print("\nSlope:", slope)
print("Intercept:", intercept)

# 6. Predict Values
y_pred = slope * X_test + intercept

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred.values)

# 7. Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 8. Visualize Regression Line
plt.scatter(X, y, color='blue', label='Actual Data')

plt.plot(
    X,
    slope * X + intercept,
    color='red',
    label='Regression Line'
)

plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.title("Linear Regression - Student Marks Prediction")

plt.legend()
plt.grid()
plt.show()
