# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import ssl
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo


def load_data(name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _handle_ssl_verification()
    if name == "parkinsons":
        return _load_parkinsons()
    if name == "tictactoe":
        return _load_tictactoe()
    if name == "splice":
        return _load_splice()
    if name == "mnist":
        return _load_mnist()
    if name == "credit":
        return _load_credit()
    if name == "mushrooms":
        raise ValueError(
            "Unable to load the data. Please obtain the variant of the mushrooms dataset with RBF kernel features from https://github.com/IssamLaradji/sps"
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _load_parkinsons() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # fetch dataset
    data = fetch_ucirepo(id=174)

    # data (as pandas dataframes)
    X = data.data.features
    y = data.data.targets

    categorical_cols = []
    df_encoded = pd.get_dummies(X, columns=categorical_cols)

    scaler = preprocessing.StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
    X = X.fillna(0)
    X = X.values

    y = np.array([float(elem) for elem in y["status"]])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, stratify=y
    )

    return X_train, X_test, y_train, y_test


def _load_tictactoe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = fetch_ucirepo(id=101)

    # data (as pandas dataframes)
    X = data.data.features
    y = data.data.targets

    categorical_cols = X.columns
    df_encoded = pd.get_dummies(X, columns=categorical_cols)

    scaler = preprocessing.StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
    X = X.fillna(0)
    X = X.values

    y = np.array([1.0 if elem == "positive" else 0.0 for elem in y["class"]])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    return X_train, X_test, y_train, y_test


def _load_mnist() -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset = fetch_openml("mnist_784", cache=True, as_frame=False)

    X = dataset["data"] / 255
    y = dataset["target"]

    is_3_or_5 = (y == "3") | (y == "5")

    X = X[is_3_or_5]
    y = y[is_3_or_5]

    y = np.array([1 if elem == "5" else 0 for elem in y])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    return X_train, X_test, y_train, y_test


def _load_splice() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = fetch_ucirepo(id=69)

    X = data.data.features
    y = data.data.targets

    is_receptor_or_donor = (y["class"] != "N").values
    y = y[is_receptor_or_donor]
    X = X.iloc[is_receptor_or_donor]

    categorical_cols = X.columns
    # One-hot encode the categorical columns
    df_encoded = pd.get_dummies(X, columns=categorical_cols)
    # Initialize a scaler
    scaler = preprocessing.StandardScaler()
    # Fit and transform the data
    X = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
    X = X.fillna(0)

    # to numpy
    X = X.values

    y = np.array([0.0 if elem == "IE" else 1.0 for elem in y["class"]])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    return X_train, X_test, y_train, y_test


def _load_credit() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # fetch dataset
    credit_approval = fetch_ucirepo(id=27)

    # data (as pandas dataframes)
    X = credit_approval.data.features
    y = credit_approval.data.targets

    categorical_cols = ["A13", "A12", "A10", "A9", "A7", "A6", "A5", "A4", "A1"]
    # One-hot encode the categorical columns
    df_encoded = pd.get_dummies(X, columns=categorical_cols)
    # Initialize a scaler
    scaler = preprocessing.StandardScaler()
    # Fit and transform the data
    X = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
    X = X.fillna(0)
    # to numpy
    X = X.values

    y = np.array([1.0 if elem == "+" else 0.0 for elem in y["A16"]])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    return X_train, X_test, y_train, y_test


def _handle_ssl_verification():
    if not os.environ.get("PYTHONHTTPSVERIFY", "") and getattr(
        ssl, "_create_unverified_context", None
    ):
        ssl._create_default_https_context = ssl._create_unverified_context
