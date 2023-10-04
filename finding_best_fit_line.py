from matplotlib import pyplot as plt
import numpy as np

# Preparing the data to be computed and plotted
dt = np.array(
    [
        [0.05, 0.11],
        [0.13, 0.14],
        [0.19, 0.17],
        [0.24, 0.21],
        [0.27, 0.24],
        [0.29, 0.32],
        [0.32, 0.30],
        [0.36, 0.39],
        [0.37, 0.42],
        [0.40, 0.40],
        [0.07, 0.09],
        [0.02, 0.04],
        [0.15, 0.19],
        [0.39, 0.32],
        [0.43, 0.48],
        [0.44, 0.41],
        [0.47, 0.49],
        [0.50, 0.57],
        [0.53, 0.59],
        [0.57, 0.51],
        [0.58, 0.60],
    ]
)

# Preparing X and y from the given data
X = dt[:, 0]
y = dt[:, 1]

# Calculating parameters (Here, intercept-theta1 and slope-theta0)
# of the line using the numpy.polyfit() function
theta = np.polyfit(X, y, 1)

print(f"The parameters of the line: {theta}")

# Now, calculating the y-axis values against x-values according to
# the parameters theta0, theta1 and theta2
y_line = theta[1] + theta[0] * X

# Plotting the data points and the best fit line
plt.scatter(X, y)
plt.plot(X, y_line, "r")
plt.title("Best fit line using numpy.polyfit()")
plt.xlabel("x-axis")
plt.ylabel("y-axis")

plt.show()
