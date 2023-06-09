from sklearn.linear_model import LinearRegression
import numpy as np

# Define your two lists of numbers
list1 = [1, 2, 3, 4, 5]
list2 = [2, 4, 6, 8, 10]

# Calculate the correlation coefficient using numpy
correlation_coefficient = np.corrcoef(list1, list2)[0, 1]

# Print the correlation coefficient
print("Correlation coefficient:", correlation_coefficient)

X = np.array(list1).reshape(-1, 1)
y = np.array(list2)

# Fit a linear regression model
regression_model = LinearRegression()
regression_model.fit(X, y)

# Predict new data points based on list1
new_data = [6, 7, 8]
predicted_values = regression_model.predict(np.array(new_data).reshape(-1, 1))

# Print the predicted values
print("Predicted values:", predicted_values)