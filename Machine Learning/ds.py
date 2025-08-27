import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset = pd.read_csv(r"C:\Users\pc\Downloads\Salary_Data.csv")


X = dataset.iloc[:, :-1]

y = dataset.iloc[:, -1]



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size= 0.2, random_state=0) 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience(Test set')
plt.xlabel('Years of Experience')
plt.ylable('Salary')
plt.show()

m = regressor.coef_

c=regressor.intercept_

(m*12) + c
(m*20) + c


bias = regressor.score

# STATS FOR SIMP LINEAR MODEL
# mean mode median
dataset.mean()

dataset['Salary'].mean()
dataset.median()

dataset['Salary'].mode()

# varriation
dataset.var()
dataset['Salary'].var()

dataset['Salary'].var()
dataset.std()

dataset['Salary'].std()


from scipy.stats import variation

variation(dataset.values)

variation(dataset['Salary'])

# correlation

dataset.corr()

dataset['Salary'].corr

dataset['Salary'].corr(dataset['YearsExperience'])
# Skewness

dataset.skew()
dataset['Salary'].skew()

#standard error

dataset.sem()
dataset['Salary'].sem()

# Z-score

import scipy.stats as stats

dataset.apply(stats.zscore)
stats.zscore(dataset['Salary'])
# Degree of freedom

a = dataset.shape[0]
b = dataset.shape[1]

degree_of_freedom = a-b

print()

# ssr

y_mean = np.mean(y)

SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

# SSE
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

# SST 

mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

# R2

r_square = 1=SSR/SST
print(r_square)
