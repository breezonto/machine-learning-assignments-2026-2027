from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import time
import numpy as np
import pandas as pd


def evaluate_feature_selection(
    X_train,
    X_test,
    y_train,
    y_test,
    selector,
    classifier,
    method_name="Feature Selection"
):
    """
    Evaluate feature selection using predictive performance,
    dimensionality reduction, and computational efficiency.
    """

    # 1. Feature selection
    selector = clone(selector)

    start_time = time.perf_counter()
    X_train_selected = selector.fit_transform(X_train, y_train)
    selection_time = time.perf_counter() - start_time

    X_test_selected = selector.transform(X_test)

    # 2. Train classifier
    model = clone(classifier)

    start_time = time.perf_counter()
    model.fit(X_train_selected, y_train)
    training_time = time.perf_counter() - start_time

    # 3. Prediction
    y_pred = model.predict(X_test_selected)

    # 4. Evaluation metrics
    results = {
        "Method": method_name,
        "Original Features": X_train.shape[1],
        "Selected Features": X_train_selected.shape[1],
        "Reduction (%)": (
            1 - X_train_selected.shape[1] / X_train.shape[1]
        ) * 100,
        "Selection Time (s)": selection_time,
        "Training Time (s)": training_time,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_test, y_pred, zero_division=0
        ),
    }

    # ROC-AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_selected)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test_selected)
    else:
        y_score = None

    results["ROC-AUC"] = (
        roc_auc_score(y_test, y_score)
        if y_score is not None else np.nan
    )

    return results, selector
