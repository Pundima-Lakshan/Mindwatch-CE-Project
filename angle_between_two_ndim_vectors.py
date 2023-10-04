import numpy as np
from matplotlib import pyplot as plt


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

    plt.quiver(*origin, V[:, 0], V[:, 1], color=["b", "g"], scale=1)
    plt.show()


def main():
    v1 = (1, 1)
    v2 = (1, 0)

    plot_two_line_vectors(v1, v2)
    angle = angle_between(v1, v2)

    print(angle)


if __name__ == "__main__":
    main()
