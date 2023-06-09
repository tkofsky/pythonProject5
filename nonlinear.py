import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Sample data
x = [1, 2, 3, 4, 5]  # List of independent variable values
y = [2.1, 4.2, 6.3, 8.4, 10.5]  # List of dependent variable values
y = [3, 9, 27, 81,243]  # List of dependent variable values
y = [2.4, 7.2, 21.6, 64.8,194.5]
y = [60.4, 50.2, 150.6, 630.45,900.5]
# Define the non-linear regression function
def nonlinear_func(x, a, b, c):

    return a * np.power(x, 2) + b * x + c

# Perform non-linear regression
popt, pcov = curve_fit(nonlinear_func, x, y)

# Generate predictions for new values of x
x_new = np.linspace(1, 5, 100)  # Generating 100 values between 1 and 5
y_pred = nonlinear_func(x_new, *popt)

# Visualize the results
plt.scatter(x, y, label='Original Data')
plt.plot(x_new, y_pred, label='Non-linear Regression', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()