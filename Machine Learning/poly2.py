import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
dataset = pd.read_csv(r"C:\Users\pc\Downloads\emp_sal.csv")
x = dataset.iloc[:, 1:2].values  # Position level
y = dataset.iloc[:, 2].values    # Salary

# Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Linear Regression Visualization
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Linear Regression Prediction
lin_model_pred = lin_reg.predict([[6]])
print("Linear Regression Prediction for level 6:", lin_model_pred)

# Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)

poly_reg2 = LinearRegression()
poly_reg2.fit(x_poly, y)

# Polynomial Regression Visualization
plt.scatter(x, y, color='red')
plt.plot(x, poly_reg2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Polynomial Regression Prediction
poly_model_pred = poly_reg2.predict(poly_reg.fit_transform([[6]]))
print("Polynomial Regression Prediction for level 6:", poly_model_pred)


