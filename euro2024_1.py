import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Prepare the data
data = {
    "Year": [1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020],
    "Total Goals": [17, 13, 7, 10, 19, 27, 41, 34, 32, 64, 85, 77, 77, 76, 108, 142]
}
df = pd.DataFrame(data)

# Reshape the data
X = df["Year"].values.reshape(-1, 1)
y = df["Total Goals"].values

# Create the model and fit it
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the data and the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Total Goals')
plt.title('Regression Analysis of Total Goals in UEFA European Championship')
plt.show()

# Print the coefficients
print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"R^2 Score: {model.score(X, y)}")
