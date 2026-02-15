# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf

# Step 1: Download historical stock data (example: Apple)
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2025-01-01")

# Step 2: Feature engineering
# We'll predict 'Close' price using 'Open', 'High', 'Low', 'Volume'
features = ["Open", "High", "Low", "Volume"]
X = data[features]
y = data["Close"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Step 7: Visualization
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label="Actual Price", color="blue")
plt.plot(y_test.index, y_pred, label="Predicted Price", color="red")
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
