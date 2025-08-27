# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 10:35:22 2025

@author: pc
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# generate some random data
np.random.seed(42)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# train a linear regression model
model = LinearRegression()
model.fit(x, y)

#Make predictions
x_new = np.array([[0],[2]])
y_pred = model.predict(x_new)

# visualize the data and the linear regression line
plt.scatter(x, y, label='Data points')
plt.plot(x_new, y_pred, 'r-',
         label='Linear Regression Line')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Example')
plt.show()