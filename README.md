# Optimal Isolation Forest for Anomaly Detection

This repository contains the code for the experiments of the paper "OptIForest: Optimal Isolation Forest for Anomaly Detection".

# Requirement

Some dependencies had newer versions than those listed in the paper; where
necessary, slight adjustments were made to ensure compatibility with Python 3.11, while preserving core functionality.

- numpy==1.20.1
- sklearn==0.22.1
- pandas==1.4.1

# Dataset

We evaluate all methods on 20 widely-used benchmark datasets, which are available in public [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php), [Kaggle Repository](https://www.kaggle.com/datasets), and [ADRepository](https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets).

# Repository Creation

The original author's repository was forked and the new branch was ads_Final_project was created, where all the works are committed.

# Experiment

### You can try different `--threshold` (the cut threshold), `--branch` (number of branching factor) to see how the AUC performance changes.

### An example for running the original code:

    python demo.py --dataset=ad --threshold=403 --branch=0

### An example for running the extension/replication code:

    python -m experiments.optiforest_ad_creditcard

## Dataset Access

The `creditcard.csv` dataset file downloaded from Kaggle(Credit Card Fraud Detection) is too large to include in this repository. The data can be manually downloaded from Kaggle. The link for download is: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud Before running the experiment, the dataset file is required to place inside the data folder of the project.
