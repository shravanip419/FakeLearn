
# EXPERIMENT 4: MULTIVARIATE LINEAR REGRESSION
# Predict house prices using multiple features

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 2. Create Dataset (10 Rows)
data = {
    'Area': [500, 600, 700, 800, 900,
             1000, 1100, 1200, 1300, 1400],

    'Bedrooms': [1, 1, 2, 2, 2,
                 3, 3, 3, 4, 4],

    'Age': [10, 8, 7, 6, 5,
            4, 3, 3, 2, 1],

    'Price': [25, 30, 35, 40, 45,
              50, 55, 60, 65, 70]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# 3. Data Preprocessing

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing rows if any
df = df.dropna()

# Display dataset information
print("\nDataset Information:")
print(df.info())

print("\nDataset Statistics:")
print(df.describe())

# 4. Feature Selection

# Independent Variables
X = df[['Area', 'Bedrooms', 'Age']]

# Dependent Variable
y = df['Price']

print("\nSelected Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

# 5. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# 6. Create Linear Regression Model
model = LinearRegression()

# 7. Train Model
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# 8. Predict Values
y_pred = model.predict(X_test)

print("\nActual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)

# 9. Display Coefficients and Intercept
print("\nRegression Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(feature, ":", coef)

print("Intercept:", model.intercept_)

# 10. Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 11. Actual vs Predicted Graph
plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred, color='blue')

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")

plt.title("Actual vs Predicted House Prices")

plt.grid()
plt.show()

# 12. Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(7, 5))

plt.scatter(y_pred, residuals, color='red')

plt.axhline(y=0, color='black', linestyle='--')

plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.grid()
plt.show()
