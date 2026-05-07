"""
Binary Bayesian Classification from Scratch

This module implements a Bayesian classifier for a binary classification problem
with two Gaussian classes sharing the same covariance matrix.

The implementation follows the theoretical discriminant function:

    h(X) = (M2 - M1)^T Sigma^-1 X
           + 1/2 (M1^T Sigma^-1 M1 - M2^T Sigma^-1 M2)

Decision rule:

    h(X) < 0 -> Class I
    h(X) > 0 -> Class II

The code is intentionally written in a clear and educational style, closely
following the original notebook implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def generate_artificial_data(mean_1, mean_2, covariance, n_samples, seed=0):
    """
    Generate artificial data for two Gaussian classes.
    """
    np.random.seed(seed)

    y1, y2 = np.random.multivariate_normal(mean_1, covariance, n_samples).T
    z1, z2 = np.random.multivariate_normal(mean_2, covariance, n_samples).T

    return y1, y2, z1, z2


def calculate_discrimination_parameters(mean_1, mean_2, covariance):
    """
    Calculate parameters used in the linear discriminant function.

    This follows the original code structure:

        invSigma = inverse of covariance matrix
        M3 = M2 - M1
        m1, m2 = components of M3

    The constant term is:

        0.5 * (M1.T * invSigma * M1 - M2.T * invSigma * M2)
    """
    invSigma = linalg.inv(covariance)

    M3 = mean_2.T - mean_1.T
    m1 = M3[0]
    m2 = M3[1]

    constant = 0.5 * (
        mean_1.T.dot(invSigma).dot(mean_1)
        - mean_2.T.dot(invSigma).dot(mean_2)
    )

    return invSigma, M3, m1, m2, constant


def calculate_discrimination_line(x1_values, invSigma, m1, m2, constant):
    """
    Calculate x2 values for the discrimination line.

    The discriminant function is:

        h(x1, x2) =
        (m1*invSigma[0,0] + m2*invSigma[1,0]) * x1
        +
        (m1*invSigma[0,1] + m2*invSigma[1,1]) * x2
        +
        constant

    The decision boundary is obtained by setting h(x1, x2) = 0
    and solving for x2.
    """
    coefficient_x1 = m1 * invSigma[0, 0] + m2 * invSigma[1, 0]
    coefficient_x2 = m1 * invSigma[0, 1] + m2 * invSigma[1, 1]

    x2_values = (-coefficient_x1 * x1_values - constant) / coefficient_x2

    return x2_values


def calculate_h_values(x1_points, x2_points, invSigma, m1, m2, constant):
    """
    Calculate values of the discriminant function h(X).

    This is the same formula used for drawing the discrimination line,
    but now applied to actual samples.
    """
    coefficient_x1 = m1 * invSigma[0, 0] + m2 * invSigma[1, 0]
    coefficient_x2 = m1 * invSigma[0, 1] + m2 * invSigma[1, 1]

    h_values = coefficient_x1 * x1_points + coefficient_x2 * x2_points + constant

    return h_values


def count_type_1_errors(h_values_class_1):
    """
    Count Type I errors.

    Type I error:
    A sample from Class I is classified as Class II.

    Since Class I corresponds to h(X) < 0,
    an error occurs when h(X) > 0.
    """
    error1 = 0

    for i in range(len(h_values_class_1)):
        if h_values_class_1[i] > 0:
            error1 = error1 + 1

    return error1


def count_type_2_errors(h_values_class_2):
    """
    Count Type II errors.

    Type II error:
    A sample from Class II is classified as Class I.

    Since Class II corresponds to h(X) > 0,
    an error occurs when h(X) < 0.
    """
    error2 = 0

    for i in range(len(h_values_class_2)):
        if h_values_class_2[i] < 0:
            error2 = error2 + 1

    return error2


def calculate_accuracy(error1, error2, n_test_samples):
    """
    Calculate classification error and accuracy.

    For equal prior probabilities:

        P = 0.5 * Eps1 + 0.5 * Eps2

    where:
        Eps1 = Type I error probability
        Eps2 = Type II error probability

    Accuracy is:

        T = (1 - P) * 100
    """
    Eps1 = error1 / n_test_samples
    Eps2 = error2 / n_test_samples

    P = 0.5 * Eps1 + 0.5 * Eps2
    T = (1 - P) * 100

    return Eps1, Eps2, P, T


def plot_two_classes(y1, y2, z1, z2, title="Two classes presented in x1-x2 space"):
    """
    Plot two classes in the x1-x2 feature space.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(y1, y2, '*m', label='Class I')
    plt.plot(z1, z2, 'oc', label='Class II')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


def plot_classes_with_discrimination_line(
    y1, y2, z1, z2, x1_line, x2_line,
    title="Training dataset with Bayesian discrimination line"
):
    """
    Plot two classes together with the discrimination line.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(y1, y2, '*m', label='Class I')
    plt.plot(z1, z2, 'oc', label='Class II')
    plt.plot(x1_line, x2_line, 'k--', label='Discrimination line')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


def plot_training_and_test_data(
    y1, y2, z1, z2,
    y11, y21, z11, z21,
    x1_line, x2_line,
    title="Training and testing dataset with discrimination line"
):
    """
    Plot training data, test data and the discrimination line.
    """
    plt.figure(figsize=(8, 6))

    plt.plot(y1, y2, '*m', label='Training Class I')
    plt.plot(z1, z2, 'oc', label='Training Class II')

    plt.plot(y11, y21, '*r', label='Test Class I')
    plt.plot(z11, z21, '*g', label='Test Class II')

    plt.plot(x1_line, x2_line, 'k--', label='Discrimination line')

    plt.grid(True)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


def run_demo():
    """
    Run the complete binary Bayesian classification demo.
    """
    M1 = np.array([-1, 4])
    M2 = np.array([3, 1])

    Sigma = np.array([
        [2, -1],
        [-1, 1]
    ])

    N = 200

    y1, y2, z1, z2 = generate_artificial_data(M1, M2, Sigma, N, seed=0)

    plot_two_classes(y1, y2, z1, z2)

    invSigma, M3, m1, m2, constant = calculate_discrimination_parameters(M1, M2, Sigma)

    x1_line = np.arange(-5, 7, 0.1)
    x2_line = calculate_discrimination_line(x1_line, invSigma, m1, m2, constant)

    plot_classes_with_discrimination_line(y1, y2, z1, z2, x1_line, x2_line)

    # New test data
    M11 = np.array([-1.1, 4.1])
    M21 = np.array([3.03, 1.02])

    Sigma1 = np.array([
        [2.1, -1.1],
        [-1.1, 1.1]
    ])

    N1 = 50

    y11, y21, z11, z21 = generate_artificial_data(M11, M21, Sigma1, N1, seed=1)

    plot_training_and_test_data(
        y1, y2, z1, z2,
        y11, y21, z11, z21,
        x1_line, x2_line
    )

    h11 = calculate_h_values(y11, y21, invSigma, m1, m2, constant)
    h21 = calculate_h_values(z11, z21, invSigma, m1, m2, constant)

    error1 = count_type_1_errors(h11)
    error2 = count_type_2_errors(h21)

    Eps1, Eps2, P, T = calculate_accuracy(error1, error2, N1)

    print("Type I errors:", error1)
    print("Type II errors:", error2)
    print("Type I error probability:", Eps1)
    print("Type II error probability:", Eps2)
    print("Total Bayesian error:", P)
    print("Classification accuracy:", T, "%")


if __name__ == "__main__":
    run_demo()
