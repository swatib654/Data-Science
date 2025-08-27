import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv(r"C:\Users\pc\Downloads\Investment.csv")

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

x = pd.get_dummies(x,dtype=int)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)