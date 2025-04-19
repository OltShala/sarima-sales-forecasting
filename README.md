# 🛒 Supermarket Sales Forecasting with SARIMA

Hey there! 👋

This project is something I worked on to explore how time series forecasting can be applied to real-world sales data. I used a SARIMA model (Seasonal AutoRegressive Integrated Moving Average) to predict daily supermarket sales. The whole process gave me a great chance to sharpen my skills in data prep, modeling, and visualization — and see how ML can actually help businesses plan better.

---

## 🔍 What’s Inside?

- Cleaned and aggregated daily sales data
- Used `auto_arima` to automatically find the best model parameters, with some manual improvements based PACF and ACF tests
- Trained a SARIMAX model to capture seasonality and trends
- Made short-term forecasts and visualized them alongside real data

---

## 🛠️ Tools & Libraries

- Python
- Pandas
- Statsmodels (SARIMAX)
- pmdarima (for auto_arima)
- Matplotlib
- Scikit-learn (for RMSE calculation)

---

## 📊 What You’ll See

- A side-by-side plot comparing actual and predicted sales with ARIMA model
- A 10-day sales forecast into the future with confidence intervals
- A clean and simple approach to forecasting time series data

---

💭 Motivation and Objective
I aimed to take a practical, hands-on approach to time series forecasting, going beyond simply running models to truly understand their impact on decision-making in the retail sector. This project provided an excellent opportunity to deepen my knowledge of SARIMA, a model I’ve found particularly effective for addressing seasonal data challenges, and allowed me to gain insights into its real-world applications for forecasting and business optimization

---

## 📁 Project Structure

- `script1.py` → The main forecasting script
- `supermarket_sales.csv` → The dataset (optional)
- `README.md` → This file

---

## 🙋 About Me

I’m Olt Shala, a student interested in data science, machine learning, and turning messy data into useful insights. I love exploring real applications of ML — and this is just one of the many projects I'm working on.

📎 [Let’s connect on LinkedIn!](www.linkedin.com/in/olt-shala-ab1276159)

---

> If you're into forecasting or data-driven problem-solving, I’d love to hear your thoughts. Feel free to fork this repo, try it out, or drop me a message!

