from sklearn.tree import DecisionTreeRegressor

# Example data
X = [[1500], [1800], [2400], [3000], [3500]]  # square footage
y = [200000, 250000, 300000, 400000, 500000]  # house prices

# Create and train the model
model = DecisionTreeRegressor(max_depth=3)
model.fit(X, y)

# Predict price for a new house
prediction = model.predict([[2600]])
print(prediction)


from sklearn.ensemble import RandomForestRegressor

# Sample data: Square footage vs. House price
X = [[1500], [1800], [2400], [3000], [3500]]
y = [2000000, 2500000, 3000000, 4000000, 5000000]

# Create and train the model
model = RandomForestRegressor()
model.fit(X, y)

# Predict price for a new house
prediction = model.predict([[2600]])
print("Predicted price:", prediction)
