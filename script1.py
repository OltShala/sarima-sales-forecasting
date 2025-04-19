import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("supermarket_sales.csv", parse_dates=["Date"])

# Prepare daily totals
df_daily = df.groupby("Date")["Total"].sum().sort_index()
df_daily = df_daily.asfreq('D').interpolate(method='time')

# Find SARIMA order
auto_model = auto_arima(df_daily, seasonal=True, m=7, d=1, D=1, stepwise=True)
order = auto_model.order
seasonal_order = auto_model.seasonal_order

# Train/test split
cutoff = int(len(df_daily) * 0.8)
train = df_daily[:cutoff]
test = df_daily[cutoff:]

# Fit model
model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
fitted = model.fit()

# Forecast
forecast = fitted.get_forecast(steps=len(test))
pred = forecast.predicted_mean

# Error
rmse = np.sqrt(mean_squared_error(test, pred))
print("RMSE:", rmse)

# Plot results
plt.plot(train, label="Train")
plt.plot(test, label="Test")
plt.plot(test.index, pred, label="Forecast", linestyle="--")
plt.fill_between(test.index, forecast.conf_int().iloc[:, 0], forecast.conf_int().iloc[:, 1], alpha=0.2)
plt.legend()
plt.title("SARIMA Forecast vs Actual")
plt.tight_layout()
plt.show()

# Forecast future 10 days
final_model = SARIMAX(df_daily, order=order, seasonal_order=seasonal_order).fit()
future_forecast = final_model.get_forecast(steps=10)
future_index = pd.date_range(df_daily.index[-1] + pd.Timedelta(days=1), periods=10)

# Plot future
plt.plot(df_daily[-30:], label="Recent History")
plt.plot(future_index, future_forecast.predicted_mean, label="Next 10 Days", linestyle='--', marker='o')
plt.fill_between(future_index,
                 future_forecast.conf_int().iloc[:, 0],
                 future_forecast.conf_int().iloc[:, 1], alpha=0.2)
plt.legend()
plt.title("10-Day Sales Forecast")
plt.tight_layout()
plt.show()
