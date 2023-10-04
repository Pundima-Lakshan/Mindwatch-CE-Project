import numpy as np
from matplotlib import pyplot as plt
import cv2


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    radAngle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.rad2deg(radAngle)


def best_fit_line(datapoints, plot=False):
    """
    Return
    0: gradient(m)
    1: intercept(c)
    """
    # Preparing X and y from the given data
    X = datapoints[:, 0]
    y = datapoints[:, 1]

    # Calculating parameters (Here, intercept-theta1 and slope-theta0)
    # of the line using the numpy.polyfit() function
    theta = np.polyfit(X, y, 1)

    print(f"The parameters of the line: {theta}")

    # Now, calculating the y-axis values against x-values according to
    # the parameters theta0, theta1 and theta2
    y_line = theta[1] + theta[0] * X

    # Plotting the data points and the best fit line
    if plot:
        fig1, ax1 = plt.subplots()
        ax1.scatter(X, y)
        ax1.plot(X, y_line, "r")

    return theta[0], theta[1]


def plot_two_line_vectors(v1, v2, xlabel="X-axis", ylabel="Y-axis"):
    x1 = v1[0]
    x2 = v2[0]

    y1 = v1[0]
    y2 = v2[0]

    V = np.array([v1, v2])
    origin = np.array([[0, 0], [0, 0]])  # origin point

    # Set the size of the cartesian plane
    plt.xlim([-1, 10])
    plt.ylim([-1, 10])

    plt.quiver(*origin, V[:, 0], V[:, 1], color=["b", "g"], scale=5)
    plt.show()


def draw_infinite_line_on_image(
    gradient, intercept, imageHeight, imageWidth, uploaded_image
):
    image = np.array(uploaded_image)

    width, height = imageWidth, imageHeight

    # Calculate the start and end points of the line based on the image size
    x1 = 0
    y1 = int(gradient * x1 + intercept)
    x2 = width - 1
    y2 = int(gradient * x2 + intercept)

    # Define the line color (BGR format) and thickness
    line_color = (255, 0, 0)  # Red color in RGB
    line_thickness = 2

    # Draw the infinite line on the image
    cv2.line(image, (x1, y1), (x2, y2), line_color, line_thickness)

    return image


def main():
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

    best_fit_line(dt, True)

    v1 = (1, 1)
    v2 = (1, 0)

    plot_two_line_vectors(v1, v2)
    angle = angle_between(v1, v2)

    print(angle)


if __name__ == "__main__":
    main()
