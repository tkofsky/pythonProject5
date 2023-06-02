import numpy as np

# Define your two lists of numbers
list1 = [1, 2, 3, 4, 5]
list2 = [2, 4, 6, 8, 10]

# Calculate the correlation coefficient using numpy
correlation_coefficient = np.corrcoef(list1, list2)[0, 1]

# Print the correlation coefficient
print("Correlation coefficient:", correlation_coefficient)