"""
Meta-classification wrapper based on scikit-learn's StackingClassifier.

The class reuses estimators exposed by BaseClassifier. If BaseClassifier.tune_model()
has been called first, its optimized estimators (stored in BaseClassifier.models)
are used as the stacking base estimators.

Example
-------
base = BaseClassifier(X=X_train, y=y_train)
base.tune_model("knn", X_train, y_train)
base.tune_model("svm", X_train, y_train)

meta = MetaClassifier(
    base_classifier=base,
    model_names=["knn", "svm", "classification_tree"],
    final_estimator="logistic_regression",
)
meta.fit(X_train, y_train)
y_pred = meta.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import issparse

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class MetaClassifier:
    """Build, train, tune, and evaluate a stacking ensemble.

    Parameters
    ----------
    base_classifier:
        An instance of BaseClassifier. Its ``models`` dictionary supplies
        the base estimators. Estimators are cloned by scikit-learn, so fitting
        the stacking model does not fit or mutate the original estimators.
    model_names:
        Names of the base estimators to include. If None, all estimators
        currently available in ``base_classifier.models`` are selected.
    final_estimator:
        Meta-classifier name or a scikit-learn estimator. Supported names:
        ``"logistic_regression"``, ``"random_forest"``,
        ``"decision_tree"``, and ``"svm"``.
    cv:
        Number of stratified folds used to generate out-of-fold predictions.
    stack_method:
        ``"auto"``, ``"predict_proba"``, ``"decision_function"``, or
        ``"predict"``. ``"auto"`` is generally a good default.
    passthrough:
        If True, append the original input features to the meta-classifier
        inputs. False uses only base-estimator predictions.
    random_state:
        Random seed.
    n_jobs:
        Number of parallel jobs used by StackingClassifier.
    """

    def __init__(
        self,
        base_classifier,
        model_names: list[str] | None = None,
        final_estimator: str | Any = "logistic_regression",
        cv: int = 5,
        stack_method: str = "auto",
        passthrough: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.base_classifier = base_classifier
        self.model_names = model_names
        self.final_estimator_name = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.passthrough = passthrough
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.estimators: list[tuple[str, Any]] = []
        self.final_estimator = self._build_final_estimator(final_estimator)
        self.model: StackingClassifier | None = None
        self.fitted_model: StackingClassifier | None = None
        self.search = None

        self._build_estimators()

    def _build_final_estimator(self, estimator):
        """Create the selected meta-classifier."""
        if not isinstance(estimator, str):
            return clone(estimator)

        seed = self.random_state
        estimators = {
            'logistic_regression': LogisticRegression(
                max_iter     = 2000,
                class_weight = 'balanced',
                random_state = seed,
            ),
            
            "random_forest": RandomForestClassifier(
                n_estimators = 300,
                class_weight = 'balanced',
                random_state = seed,
                n_jobs=self.n_jobs,
            ),
            
            "decision_tree": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=seed,
            ),

            "svm": SVC(
                probability=True,
                class_weight="balanced",
                random_state=seed,
            ),

            
        }

        if estimator not in estimators:
            raise ValueError(
                f"Unknown final_estimator '{estimator}'. "
                f"Choose from: {list(estimators)}."
            )
        return estimators[estimator]

    def _build_estimators(self) -> None:
        """Collect the requested base estimators from BaseClassifier."""
        available = self.base_classifier.get_models()

        names = (
            list(available)
            if self.model_names is None
            else list(self.model_names)
        )

        if len(names) < 2:
            raise ValueError(
                "Stacking requires at least two base estimators."
            )

        unknown = [name for name in names if name not in available]
        if unknown:
            raise ValueError(
                f"Unknown base estimator(s): {unknown}. "
                f"Available estimators: {list(available)}."
            )

        self.estimators = [
            (name, clone(available[name]))
            for name in names
        ]

    @staticmethod
    def _validate_data(X, y):
        """Validate feature and target arrays while preserving sparse input."""
        X = X if issparse(X) else np.asarray(X)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional; got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have different sample counts: "
                f"{X.shape[0]} vs {y.shape[0]}."
            )
        if X.shape[0] == 0:
            raise ValueError("The dataset is empty.")

        values = X.data if issparse(X) else X
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("X must contain numeric values.")
        if not np.isfinite(values).all():
            raise ValueError("X contains NaN or infinite values.")

        return X, y

    def _build_model(self, final_estimator=None) -> StackingClassifier:
        """Construct the StackingClassifier."""
        final = (
            self.final_estimator
            if final_estimator is None
            else final_estimator
        )

        return StackingClassifier(
            estimators=self.estimators,
            final_estimator=clone(final),
            cv=StratifiedKFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state,
            ),
            stack_method=self.stack_method,
            passthrough=self.passthrough,
            n_jobs=self.n_jobs,
        )

    def fit(
        self,
        X_train=None,
        y_train=None,
    ) -> StackingClassifier:
        """Fit the stacking ensemble using out-of-fold base predictions."""
        if X_train is None:
            X_train = self.base_classifier.X
        if y_train is None:
            y_train = self.base_classifier.y

        X_train, y_train = self._validate_data(X_train, y_train)

        # Rebuild to pick up any estimators tuned after this object was created.
        available = self.base_classifier.get_models()
        self.estimators = [
            (name, clone(available[name]))
            for name, _ in self.estimators
        ]

        self.model = self._build_model()
        self.model.fit(X_train, y_train)
        self.fitted_model = self.model
        return self.fitted_model

    def predict(self, X) -> np.ndarray:
        """Predict class labels with the fitted stacking ensemble."""
        if self.fitted_model is None:
            raise ValueError("Fit the meta-classifier before calling predict().")
        return self.fitted_model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities, if supported by the meta-classifier."""
        if self.fitted_model is None:
            raise ValueError(
                "Fit the meta-classifier before calling predict_proba()."
            )
        if not hasattr(self.fitted_model, "predict_proba"):
            raise AttributeError(
                "The configured stacking estimator does not support predict_proba."
            )
        return self.fitted_model.predict_proba(X)

    def tune_model(
        self,
        X_train,
        y_train,
        param_distributions: dict | None = None,
        scoring: str = "f1_macro",
        n_iter: int = 20,
        cv: int = 3,
    ) -> dict[str, Any]:
        """Tune the meta-classifier while keeping the base estimators fixed.

        The search parameters use the ``final_estimator__`` prefix.
        """
        X_train, y_train = self._validate_data(X_train, y_train)

        if param_distributions is None:
            if self.final_estimator_name == "logistic_regression":
                param_distributions = {
                    "final_estimator__C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "final_estimator__class_weight": [None, "balanced"],
                }
            elif self.final_estimator_name == "random_forest":
                param_distributions = {
                    "final_estimator__n_estimators": [100, 200, 300, 500],
                    "final_estimator__max_depth": [None, 5, 10, 20],
                    "final_estimator__min_samples_leaf": [1, 2, 5, 10],
                    "final_estimator__max_features": ["sqrt", "log2", None],
                }
            elif self.final_estimator_name == "decision_tree":
                param_distributions = {
                    "final_estimator__max_depth": [None, 3, 5, 10, 20],
                    "final_estimator__min_samples_leaf": [1, 2, 5, 10],
                    "final_estimator__class_weight": [None, "balanced"],
                }
            elif self.final_estimator_name == "svm":
                param_distributions = {
                    "final_estimator__C": [0.01, 0.1, 1, 10, 100],
                    "final_estimator__kernel": ["linear", "rbf"],
                    "final_estimator__class_weight": [None, "balanced"],
                }
            else:
                raise ValueError(
                    "Provide param_distributions when using a custom "
                    "final_estimator."
                )

        search = RandomizedSearchCV(
            estimator=self._build_model(),
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=StratifiedKFold(
                n_splits=cv,
                shuffle=True,
                random_state=self.random_state,
            ),
            random_state = self.random_state,
            n_jobs       = self.n_jobs,
            verbose      = 1,
            refit        = True,
        )
        search.fit(X_train, y_train)

        self.search = search
        self.model = search.best_estimator_
        self.fitted_model = search.best_estimator_
        self.final_estimator = clone(
            self.fitted_model.final_estimator
        )

        return {
            "model": "stacking",
            "best_score": search.best_score_,
            "best_params": search.best_params_,
            "best_estimator": search.best_estimator_,
        }

    def evaluate(self, X_test, y_test):
        y_test = np.asarray(y_test).ravel()
        y_pred = self.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
            "y_test": y_test,
            "y_pred": y_pred,
            "estimator": self.fitted_model
        }

    def fit_evaluate(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> dict[str, Any]:
        """Fit on the supplied training split and evaluate on the test split."""
        self.fit(X_train, y_train)
        return self.evaluate(X_test, y_test)

    def summary(self) -> str:
        """Return a compact description of the stacking configuration."""
        base_names = [name for name, _ in self.estimators]
        final_name = (
            self.final_estimator_name
            if isinstance(self.final_estimator_name, str)
            else self.final_estimator_name.__class__.__name__
        )
        return (
            f"Meta-classifier: StackingClassifier\n"
            f"Base classifiers: {', '.join(base_names)}\n"
            f"Final estimator: {final_name}\n"
            f"CV folds: {self.cv}\n"
            f"Stack method: {self.stack_method}\n"
            f"Passthrough: {self.passthrough}"
        )
