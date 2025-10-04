# DL4CV - Deep Learning for Computer Vision

This project implements various classical machine learning models for digit classification using the digits dataset from scikit-learn. The goal is to compare different algorithms including Logistic Regression, Linear SVM, K-Nearest Neighbors, Decision Tree, Random Forest, SGD Classifier, and Perceptron.

## Project Structure

```
DL4CV/
├─ data/                         # Data directory
├─ results/
│  ├─ figures/                   # Generated figures
│  └─ metrics/                   # Metrics JSON files
├─ src/
│  ├─ __init__.py                # Package init
│  ├─ load_data.py               # Data loading functions
│  ├─ evaluate.py                # Model evaluation functions
│  ├─ visualize.py               # Visualization functions
│  └─ models.py                  # Model constants and mappings
├─ notebooks/
│  ├─ project_overview.ipynb     # Project overview
│  ├─ part1_logistic_regression.ipynb
│  ├─ part2_linear_svm.ipynb
│  ├─ part3_knn.ipynb
│  ├─ part4_decision_tree.ipynb
│  ├─ part5_random_forest.ipynb
│  ├─ part6_sgd.ipynb
│  ├─ part7_perceptron.ipynb
│  └─ part8_comparison.ipynb
├─ requirements.txt              # Project dependencies
└─ README.md                     # This file
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebooks in the `notebooks/` directory to train and evaluate models.

## Models

The project compares the following models:

- Logistic Regression
- Linear SVM
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- SGD Classifier
- Perceptron

Each model has its own notebook with hyperparameter tuning and evaluation.

## Results

Results are saved in the `results/` directory:
- Metrics are saved as JSON files in `results/metrics/`
- Figures are saved as PNG files in `results/figures/`
