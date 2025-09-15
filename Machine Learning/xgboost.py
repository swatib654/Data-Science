#XGBoost

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv(r"C:\Users\pc\Downloads\Churn_Modelling.csv")
X = dataset.iloc[:,3: -1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)
