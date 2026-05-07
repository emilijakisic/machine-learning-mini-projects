"""
Bayesian Classification on the Iris Dataset

This module implements Bayesian classification for the Iris dataset
using petal length and petal width features.

The implementation follows the theoretical quadratic discriminant function
for Gaussian classes with different covariance matrices.

The code intentionally mirrors the notebook implementation and avoids
high-level machine learning abstractions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn import datasets


def load_iris_dataset():
    """
    Load the Iris dataset.

    Returns
    -------
    np.ndarray
        Feature matrix.
    np.ndarray
        Target labels.
    """
    dataset = datasets.load_iris()

    X = dataset.data
    Y = dataset.target

    return X, Y


def plot_iris_petal_features(X):
    """
    Plot Iris dataset using petal length and petal width.
    """
    plt.figure(figsize=(8, 6))

    plt.scatter(
        X[0:50, 2],
        X[0:50, 3],
        c='y',
        marker='*',
        label='Iris Setosa'
    )

    plt.scatter(
        X[50:100, 2],
        X[50:100, 3],
        c='r',
        marker='^',
        label='Iris Versicolor'
    )

    plt.scatter(
        X[100:150, 2],
        X[100:150, 3],
        c='g',
        marker='o',
        label='Iris Virginica'
    )

    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.title('Iris dataset using petal features')

    plt.legend()
    plt.grid(True)
    plt.show()


def plot_iris_sepal_features(X):
    """
    Plot Iris dataset using sepal length and sepal width.
    """
    plt.figure(figsize=(8, 6))

    plt.scatter(
        X[0:50, 0],
        X[0:50, 1],
        c='y',
        marker='*',
        label='Iris Setosa'
    )

    plt.scatter(
        X[50:100, 0],
        X[50:100, 1],
        c='r',
        marker='^',
        label='Iris Versicolor'
    )

    plt.scatter(
        X[100:150, 0],
        X[100:150, 1],
        c='g',
        marker='o',
        label='Iris Virginica'
    )

    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Iris dataset using sepal features')

    plt.legend()
    plt.grid(True)
    plt.show()


def prepare_iris_classes(X):
    """
    Prepare Iris classes using petal features.

    Returns
    -------
    tuple
        X1, X2, X3
    """
    X1 = X[0:50, 2:4]
    X2 = X[50:100, 2:4]
    X3 = X[100:150, 2:4]

    return X1, X2, X3


def calculate_class_statistics(X1, X2, X3):
    """
    Calculate mean vectors and covariance matrices for all classes.
    """
    sigma1 = np.cov(X1.T)
    sigma2 = np.cov(X2.T)
    sigma3 = np.cov(X3.T)

    invSigma1 = linalg.inv(sigma1)
    invSigma2 = linalg.inv(sigma2)
    invSigma3 = linalg.inv(sigma3)

    M1 = np.mean(X1, axis=0)
    M2 = np.mean(X2, axis=0)
    M3 = np.mean(X3, axis=0)

    return (
        sigma1, sigma2, sigma3,
        invSigma1, invSigma2, invSigma3,
        M1, M2, M3
    )


def calculate_qda_discriminant(
    x1,
    x2,
    mean_a,
    mean_b,
    inv_covariance_a,
    inv_covariance_b,
    covariance_a,
    covariance_b
):
    """
    Calculate the quadratic discriminant function.

    This implementation intentionally follows the original code
    and directly mirrors the theoretical equation.
    """

    term_a = 0.5 * (
        (
            (x1 - mean_a[0]) * inv_covariance_a[0][0]
            + (x2 - mean_a[1]) * inv_covariance_a[1][0]
        ) * (x1 - mean_a[0])
        +
        (
            (x1 - mean_a[0]) * inv_covariance_a[0][1]
            + (x2 - mean_a[1]) * inv_covariance_a[1][1]
        ) * (x2 - mean_a[1])
    )

    term_b = 0.5 * (
        (
            (x1 - mean_b[0]) * inv_covariance_b[0][0]
            + (x2 - mean_b[1]) * inv_covariance_b[1][0]
        ) * (x1 - mean_b[0])
        +
        (
            (x1 - mean_b[0]) * inv_covariance_b[0][1]
            + (x2 - mean_b[1]) * inv_covariance_b[1][1]
        ) * (x2 - mean_b[1])
    )

    log_term = 0.5 * np.log(
        np.linalg.det(covariance_a)
        /
        np.linalg.det(covariance_b)
    )

    h = term_a - term_b + log_term

    return h


def create_decision_grid():
    """
    Create meshgrid for decision boundary visualization.
    """
    Xaxis = np.arange(0, 8, 0.1)
    Yaxis = np.arange(0, 3, 0.1)

    x1_grid, x2_grid = np.meshgrid(Xaxis, Yaxis)

    return x1_grid, x2_grid


def plot_qda_boundaries(
    X,
    x1_grid,
    x2_grid,
    h_setosa_versicolor,
    h_versicolor_virginica
):
    """
    Plot Iris classes together with Bayesian decision boundaries.
    """
    plt.figure(figsize=(8, 6))

    plt.scatter(
        X[0:50, 2],
        X[0:50, 3],
        c='y',
        marker='*',
        label='Setosa'
    )

    plt.scatter(
        X[50:100, 2],
        X[50:100, 3],
        c='r',
        marker='^',
        label='Versicolor'
    )

    plt.scatter(
        X[100:150, 2],
        X[100:150, 3],
        c='g',
        marker='o',
        label='Virginica'
    )

    plt.contour(
        x1_grid,
        x2_grid,
        h_setosa_versicolor,
        levels=[0],
        colors='blue'
    )

    plt.contour(
        x1_grid,
        x2_grid,
        h_versicolor_virginica,
        levels=[0],
        colors='red'
    )

    plt.xlabel('Petal length')
    plt.ylabel('Petal width')

    plt.title('Iris species with Bayesian classification boundaries')

    plt.legend()
    plt.grid(True)
    plt.show()


def count_qda_errors(h_values_class_1, h_values_class_2):
    """
    Count classification errors for two classes.
    """
    error1 = 0

    for value in h_values_class_1:
        if value > 0:
            error1 = error1 + 1

    error2 = 0

    for value in h_values_class_2:
        if value < 0:
            error2 = error2 + 1

    return error1, error2


def calculate_accuracy(error1, error2, n_samples):
    """
    Calculate Bayesian classification accuracy.
    """
    Eps1 = error1 / n_samples
    Eps2 = error2 / n_samples

    P = 0.5 * Eps1 + 0.5 * Eps2

    T = (1 - P) * 100

    return Eps1, Eps2, P, T


def evaluate_setosa_vs_versicolor(
    X,
    M1,
    M2,
    invSigma1,
    invSigma2,
    sigma1,
    sigma2
):
    """
    Evaluate Setosa vs Versicolor classification.
    """

    x1p1 = X[0:50, 2]
    x2p1 = X[0:50, 3]

    h1 = calculate_qda_discriminant(
        x1p1,
        x2p1,
        M1,
        M2,
        invSigma1,
        invSigma2,
        sigma1,
        sigma2
    )

    x1p2 = X[50:100, 2]
    x2p2 = X[50:100, 3]

    h2 = calculate_qda_discriminant(
        x1p2,
        x2p2,
        M1,
        M2,
        invSigma1,
        invSigma2,
        sigma1,
        sigma2
    )

    error1, error2 = count_qda_errors(h1, h2)

    Eps1, Eps2, P, T = calculate_accuracy(error1, error2, 50)

    print('Classification of Setosa and Versicolor')
    print('Classification accuracy:', T, '%')
    print('Number of misclassified Setosa samples:', error1)
    print('Number of misclassified Versicolor samples:', error2)


def evaluate_versicolor_vs_virginica(
    X,
    M2,
    M3,
    invSigma2,
    invSigma3,
    sigma2,
    sigma3
):
    """
    Evaluate Versicolor vs Virginica classification.
    """

    x1p1 = X[50:100, 2]
    x2p1 = X[50:100, 3]

    h1 = calculate_qda_discriminant(
        x1p1,
        x2p1,
        M2,
        M3,
        invSigma2,
        invSigma3,
        sigma2,
        sigma3
    )

    x1p2 = X[100:150, 2]
    x2p2 = X[100:150, 3]

    h2 = calculate_qda_discriminant(
        x1p2,
        x2p2,
        M2,
        M3,
        invSigma2,
        invSigma3,
        sigma2,
        sigma3
    )

    error1, error2 = count_qda_errors(h1, h2)

    Eps1, Eps2, P, T = calculate_accuracy(error1, error2, 50)

    print('Classification of Versicolor and Virginica')
    print('Classification accuracy:', T, '%')
    print('Number of misclassified Versicolor samples:', error1)
    print('Number of misclassified Virginica samples:', error2)


def run_demo():
    """
    Run the complete Iris Bayesian classification demo.
    """

    X, Y = load_iris_dataset()

    # Visualization
    plot_iris_petal_features(X)
    plot_iris_sepal_features(X)

    # Prepare classes
    X1, X2, X3 = prepare_iris_classes(X)

    (
        sigma1, sigma2, sigma3,
        invSigma1, invSigma2, invSigma3,
        M1, M2, M3
    ) = calculate_class_statistics(X1, X2, X3)

    # Decision boundaries
    x1_grid, x2_grid = create_decision_grid()

    h_setosa_versicolor = calculate_qda_discriminant(
        x1_grid,
        x2_grid,
        M1,
        M2,
        invSigma1,
        invSigma2,
        sigma1,
        sigma2
    )

    h_versicolor_virginica = calculate_qda_discriminant(
        x1_grid,
        x2_grid,
        M2,
        M3,
        invSigma2,
        invSigma3,
        sigma2,
        sigma3
    )

    plot_qda_boundaries(
        X,
        x1_grid,
        x2_grid,
        h_setosa_versicolor,
        h_versicolor_virginica
    )

    # Accuracy evaluation
    evaluate_setosa_vs_versicolor(
        X,
        M1,
        M2,
        invSigma1,
        invSigma2,
        sigma1,
        sigma2
    )

    print()

    evaluate_versicolor_vs_virginica(
        X,
        M2,
        M3,
        invSigma2,
        invSigma3,
        sigma2,
        sigma3
    )


if __name__ == "__main__":
    run_demo()
