# Bayesian Classification

Implementation of Bayesian classifiers for:
- artificially generated data
- Iris dataset classification

This project demonstrates:
- Bayesian minimal error classification
- discrimination functions
- linear and quadratic decision boundaries
- Type I and Type II errors
- covariance matrix influence
- feature selection in machine learning

---

# Project Structure

```text
project-01-bayesian-classification/

├── images/
├── notebook/
├── src/
├── theory/
│   └── theory.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Technologies

- Python
- NumPy
- Matplotlib
- SciPy
- Jupyter Notebook

---

# First Classification Problem

Binary classification of artificially generated Gaussian data.

The classifier is based on:
- Bayesian minimal error decision rule
- equal covariance matrices
- linear discrimination function

## Generated Training Dataset

<p align="center">
  <img src="images/First_problem_classification_train.png" width="700"/>
</p>

## Testing Dataset with Decision Boundary

<p align="center">
  <img src="images/First_problem_classification_test.png" width="700"/>
</p>

The generated classes overlap in feature space, therefore classification errors are expected.

---

# Second Classification Problem

Classification of Iris plant species using Bayesian classifiers.

The Iris dataset contains three flower species:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

Classification is performed using:
- petal length
- petal width
- sepal length
- sepal width

## Iris Dataset

<p align="center">
  <img src="images/Iris.png" width="700"/>
</p>

---

# Feature Selection

## Petal Features

Petal features provide very good class separation.

<p align="center">
  <img src="images/features1.png" width="700"/>
</p>

## Sepal Features

Sepal features produce overlap between classes.

<p align="center">
  <img src="images/features2.png" width="700"/>
</p>

This demonstrates importance of feature selection in machine learning.

---

# Bayesian Classification of Iris Dataset

<p align="center">
  <img src="images/Second_problem_classification.png" width="700"/>
</p>

The project demonstrates:
- quadratic discrimination boundaries
- overlap between classes
- covariance matrix influence on classification

---

# Theory

Complete mathematical derivations and explanations are available in:

`theory/theory.md`

---

# Running the Project

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run Jupyter Notebook

```bash
jupyter notebook
```

---

# Results

The project shows that:
- equal covariance matrices produce linear decision boundaries
- different covariance matrices produce quadratic boundaries
- overlapping classes increase classification error
- feature selection strongly influences classification performance
- Bayesian classifiers provide strong theoretical and practical foundation for pattern recognition problems

---

# Author

Emilija Kisic
