import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Prepare the data for regression analysis
data_matches = {
    "Year": [1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024],
    "Matches": [4, 4, 5, 4, 4, 14, 15, 15, 15, 31, 31, 31, 31, 31, 51, 51, 51],
    "Total Goals": [17, 13, 7, 10, 19, 27, 41, 34, 32, 64, 85, 77, 77, 76, 108, 142, 117]
}
df_matches = pd.DataFrame(data_matches)

# Reshape the data
X_matches = df_matches["Matches"].values.reshape(-1, 1)
y_goals = df_matches["Total Goals"].values

# Create the model and fit it
model_matches = LinearRegression()
model_matches.fit(X_matches, y_goals)

# Make predictions
y_pred_matches = model_matches.predict(X_matches)

# Plot the data and the regression line
plt.scatter(X_matches, y_goals, color='blue')
plt.plot(X_matches, y_pred_matches, color='red', linewidth=2)
plt.xlabel('Number of Matches')
plt.ylabel('Total Goals')
plt.title('Regression Analysis of Total Goals vs. Number of Matches in UEFA European Championship')
plt.show()

# Print the coefficients
coefficient_matches = model_matches.coef_[0]
intercept_matches = model_matches.intercept_
r2_score_matches = model_matches.score(X_matches, y_goals)

print(f"Coefficient: {coefficient_matches}")
print(f"Intercept: {intercept_matches}")
print(f"R^2 Score: {r2_score_matches}")

# Function to predict total goals based on user input for number of matches
def predict_goals(num_matches):
    matches = np.array([[num_matches]])
    predicted_goals = model_matches.predict(matches)
    return predicted_goals[0]

# Take user input for number of matches
num_matches = int(input("Enter the number of matches: "))
predicted_goals = predict_goals(num_matches)
print(f"Predicted Total Goals for {num_matches} matches: {predicted_goals}")