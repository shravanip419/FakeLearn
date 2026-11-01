import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Create dataset
data = {
    'Area': [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
    'Bedrooms': [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    'Age': [10, 8, 7, 6, 5, 4, 3, 3, 2, 1],
    'Price': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# 2. Separate X and y
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate model
print("\nActual Prices:", list(y_test))
print("Predicted Prices:", y_pred)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# 7. Display coefficients and intercept
print("\nCoefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 8. Predict for new data
new_data = pd.DataFrame({
    'Area': [1000],
    'Bedrooms': [3],
    'Age': [5]
})

prediction = model.predict(new_data)

print("Predicted House Price:", prediction[0])

# 9. Visualization - Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue')

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

plt.show()

# 10. Visualization - Residual Plot
residuals = y_test - y_pred

plt.scatter(y_pred, residuals, color='red')
plt.axhline(y=0, color='black', linestyle='--')

plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.show()