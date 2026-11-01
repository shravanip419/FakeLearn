
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Create dataset
df = pd.DataFrame({
    "Area": [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
    "Price": [20, 24, 27, 33, 35, 41, 43, 49, 51, 57]
})

print("Dataset:\n", df)

# 2. Split features and target
X = df[["Area"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Apply Linear Regression
result = linregress(X_train["Area"], y_train)

slope = result.slope
intercept = result.intercept

print("\nSlope:", slope)
print("Intercept:", intercept)

# 4. Predict on test data
y_pred = slope * X_test["Area"] + intercept

# Print actual and predicted values separately
print("\nActual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred.values.round(2))

# 5. Evaluate model
print("\nModel Evaluation:")
print("R2 Score:", round(r2_score(y_test, y_pred), 4))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 4))
print("MSE:", round(mean_squared_error(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))

# 6. Predict price for a new house
area = float(input("\nEnter house area in sq.ft.: "))
price = slope * area + intercept

print("Predicted price:", round(price, 2), "lakh")

# 7. Plot regression line
plt.scatter(X_train["Area"], y_train, label="Training Data")
plt.scatter(X_test["Area"], y_test, label="Testing Data")

x_line = np.linspace(df["Area"].min(), df["Area"].max(), 100)
y_line = slope * x_line + intercept

plt.plot(x_line, y_line, color="red", label="Regression Line")
plt.xlabel("Area (sq.ft.)")
plt.ylabel("Price (lakh)")
plt.title("Linear Regression: House Price Prediction")
plt.legend()
plt.tight_layout()
plt.show()
