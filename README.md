### Name: KARTHIKEYAN R
### Register No: 212222240046
### Date:
# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
1.Import necessary libraries (NumPy, Matplotlib)

2.Load the dataset

3.Calculate the linear trend values using least square method

4.Calculate the polynomial trend values using least square method

5.End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv('/content/Gold Price.csv')
df.head(5)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
# Convert 'Date' column to datetime, handling potential errors
df['Date'] = pd.to_datetime(df['Date'], errors='coerce') 

# Drop rows with invalid dates (if any)
df = df.dropna(subset=['Date'])

# Now apply the toordinal() function
df['Date_ordinal'] = df['Date'].apply(lambda x: x.toordinal())
X = df['Date_ordinal'].values.reshape(-1, 1)
print(df.columns) 
```
### A - LINEAR TREND ESTIMATION
```
y = df['Price'].values  
if 'Close' in df.columns:
    y = df['Close'].values
linear_model = LinearRegression()
linear_model.fit(X, y)
df['Linear_Trend'] = linear_model.predict(X)
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'], label='Original Data', color='blue')  
plt.plot(df['Date'], df['Linear_Trend'], color='yellow', label='Linear Trend')
plt.title('Linear Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
```

### B- POLYNOMIAL TREND ESTIMATION
```
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
df['Polynomial_Trend'] = poly_model.predict(X_poly)
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'], label='Original Data', color='blue') # Changed 'price' to 'Price'
plt.plot(df['Date'], df['Polynomial_Trend'], color='green', label='Polynomial Trend (Degree 2)')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

```

### OUTPUT
![image](https://github.com/user-attachments/assets/6204da06-a24d-41c8-9875-c3d47131568d)

### A - LINEAR TREND ESTIMATION
![Untitled](https://github.com/user-attachments/assets/dea23d9c-7a30-4849-9462-0e911cb96bbd)

### B- POLYNOMIAL TREND ESTIMATION
![Untitled](https://github.com/user-attachments/assets/458667c0-1051-418a-b025-405c38bc9437)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
