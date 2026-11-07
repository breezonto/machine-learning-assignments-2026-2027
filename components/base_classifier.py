"""
base_classifier.py

A small scikit-learn wrapper for the classification algorithms used in the
practical application.

Expected NPZ format (default keys):
    X: feature matrix, shape (n_samples, n_features)
    y: class labels, shape (n_samples,)

Example:
    from base_classifier import BaseClassifier

    models = BaseClassifier.from_npz("census_income.npz")
    results = models.fit_evaluate(test_size=0.2, random_state=42)

    for name, result in results.items():
        print(name, result["accuracy"])

You can change the NPZ keys with x_key and y_key. The NPZ file should contain
numeric features. Encode categorical/string features before saving it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold    
)

from sklearn.pipeline        import Pipeline
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.tree            import DecisionTreeClassifier, export_text
from sklearn.svm             import SVC
from sklearn.linear_model    import LogisticRegression
from sklearn.naive_bayes     import CategoricalNB, GaussianNB
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing   import FunctionTransformer
from wittgenstein            import RIPPER


class SklearnRIPPER(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        k=2,
        prune_size=0.33,
        dl_allowance=64,
        n_discretize_bins=10,
        random_state=42,
        verbosity=0,
    ):
        self.k = k
        self.prune_size = prune_size
        self.dl_allowance = dl_allowance
        self.n_discretize_bins = n_discretize_bins
        self.random_state = random_state
        self.verbosity = verbosity

    def fit(self, X, y):
        self.estimator_ = RIPPER(
            k=self.k,
            prune_size=self.prune_size,
            dl_allowance=self.dl_allowance,
            n_discretize_bins=self.n_discretize_bins,
            random_state=self.random_state,
            verbosity=self.verbosity,
        )

        self.estimator_.fit(X, y)
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        return np.asarray(self.estimator_.predict(X))


class BaseClassifier:
    """
    Load an NPZ dataset and provide a consistent interface to several
    scikit-learn classification estimators.

    Parameters
    ----------
    X:
        Two-dimensional numeric feature array.
    y:
        One-dimensional target-label array.
    random_state:
        Random seed used by estimators and train/test splitting.
    """

    def __init__(
        self,
        X            : np.ndarray,
        y            : np.ndarray,
        random_state : int = 42,
    ) -> None:
        self.X = X if issparse(X) else np.asarray(X)
        self.y = np.asarray(y).ravel()
        self.random_state = random_state

        if self.X.ndim != 2:
            raise ValueError(
                f"X must be 2-dimensional; got shape {self.X.shape}."
            )

        if self.y.ndim != 1:
            raise ValueError("y must be 1-dimensional.")

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X and y have different sample counts: "
                f"{self.X.shape[0]} vs {self.y.shape[0]}."
            )

        if self.X.shape[0] == 0:
            raise ValueError("The dataset is empty.")

        if not np.issubdtype(self.X.dtype, np.number):
            raise TypeError("X must contain numeric values.")

        if issparse(self.X):
            valid = np.isfinite(self.X.data).all()
        else:
            valid = np.isfinite(self.X).all()

        if not valid:
            raise ValueError("X contains NaN or infinite values.")

        self.models: dict[str, BaseEstimator] = {}
        self.fitted_models: dict[str, BaseEstimator] = {}
        self._build_models()

    @classmethod
    def from_npz(
        cls,
        file_path    : str | Path,
        x_key        : str = "X",
        y_key        : str = "y",
        random_state : int = 42,
    ) -> "BaseClassifier":
        """Create a classifier wrapper from arrays stored in an NPZ file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {file_path}")

        with np.load(file_path, allow_pickle=False) as data:
            available_keys = list(data.files)
            if x_key not in data or y_key not in data:
                raise KeyError(
                    f"Could not find keys {x_key!r} and/or {y_key!r} in {file_path}. "
                    f"Available keys: {available_keys}"
                )
            X = data[x_key]
            y = data[y_key]

        return cls(X=X, y=y, random_state=random_state)

    def _build_models(self) -> None:
        """Initialize all supported estimators."""
        seed = self.random_state

        # Distance- and gradient-based models benefit from feature scaling.
        self.models = {
            # "knn": Pipeline([
            #     ("scaler", StandardScaler(with_mean=False)),
            #     ("classifier", KNeighborsClassifier()),
            # ]),

            # "classification_tree": DecisionTreeClassifier(
            #     class_weight="balanced",
            #     random_state=seed,
            # ),

            # "rule_induction": Pipeline([
            #     ("to_dense", FunctionTransformer(
            #         lambda X: X.toarray() if issparse(X) else X
            #     )),
            #     ("classifier", SklearnRIPPER(
            #         k=2,
            #         prune_size=0.33,
            #         dl_allowance=64,
            #         n_discretize_bins=10,
            #         random_state=seed,
            #         verbosity=0,
            #     )),
            # ]),

            "logistic_regression": Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("classifier", LogisticRegression(
                    # max_iter     = 2000,
                    # random_state = seed,
                    
                    solver="liblinear",
                    penalty="l1",
                    class_weight={0: 1, 1: 3},
                    C=0.01,
                    max_iter=2000,
                    random_state=seed,
                )),
            ]),

            # "svm": Pipeline([
            #     ("scaler", StandardScaler(with_mean=False)),
            #     ("classifier", SVC(
            #         probability=True,
            #         random_state=seed,
            #     )),
            # ]),

            ## "Discriminant Analysis" is the standard name (not "Determinant Analysis"). 
            ## Both common variants are included.
            
            # "linear_discriminant_analysis": Pipeline([
            #     ("to_dense", FunctionTransformer(
            #         lambda X: X.toarray() if hasattr(X, "toarray") else X
            #     )),
            #     ("classifier", LinearDiscriminantAnalysis(
            #         solver="lsqr",
            #         shrinkage="auto",
            #     )),
            # ]),
            

            # "quadratic_discriminant_analysis": Pipeline([
            #      ("to_dense", FunctionTransformer(
            #         lambda X: X.toarray() if hasattr(X, "toarray") else X
            #     )),
            #     ("classifier", QuadraticDiscriminantAnalysis(reg_param=0.1)),
            # ]),

            ## CategoricalNB expects non-negative integer category codes.
            # "bayesian_discrete": Pipeline([
            #     ("to_dense", FunctionTransformer(
            #         lambda X: X.toarray() if hasattr(X, "toarray") else X
            #     )),
            #     ("classifier", CategoricalNB()),
            # ]),

            # GaussianNB is the usual continuous-feature Naive Bayes model.
            #"bayesian_continuous": Pipeline([
            #    ("to_dense", FunctionTransformer(
            #         lambda X: X.toarray() if hasattr(X, "toarray") else X
            #     )),
            #     ("classifier", GaussianNB()),
            # ]),

            # "artificial_neural_network": Pipeline([
            #     ("scaler", StandardScaler(with_mean=False)),
            #     ("classifier", MLPClassifier(
            #         hidden_layer_sizes=(100,),
            #         max_iter=500,
            #         early_stopping=True,
            #         random_state=seed,
            #    )),
            # ]),
        }

        # Hyperparameter search spaces
        self.param_grids = {
            "knn": {
                "classifier__n_neighbors": [3, 5, 7, 9, 11, 15],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"],
            },

            "classification_tree": {
                "max_depth": [None, 5, 10, 15, 20, 30],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "criterion": ["gini", "entropy", "log_loss"],
                "class_weight": ["balanced", "balanced_subsample", None],
            },

            "rule_induction": {
                "classifier__k": [1, 2, 3, 5],
                "classifier__prune_size": [0.25, 0.33, 0.4],
                "classifier__dl_allowance": [32, 64, 128],
                "classifier__n_discretize_bins": [5, 10, 20],
            },

            "logistic_regression": {
                "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "classifier__penalty": ["l1", "l2"],
                "classifier__solver": ["liblinear"],
                "classifier__class_weight": [
                    None,
                    "balanced",
                    {0: 1, 1: 2},
                    {0: 1, 1: 3},
                    {0: 1, 1: 5},
                    {0: 1, 1: 10},
                ],
            },

            "svm": {
                "classifier__C": [0.01, 0.1, 1, 10, 100],
                "classifier__kernel": ["linear", "rbf"],
                "classifier__gamma": ["scale", "auto", 0.01, 0.1, 1],
                "classifier__class_weight": [
                    None,
                    "balanced",
                    {0: 1, 1: 2},
                    {0: 1, 1: 3},
                    {0: 1, 1: 5},
                    {0: 1, 1: 10},
                ],
            },

            "linear_discriminant_analysis": [
                {
                    "classifier__solver": ["lsqr"],
                    "classifier__shrinkage": [
                        "auto", 0.1, 0.3, 0.5, 0.7
                    ],
                },
                {
                    "classifier__solver": ["eigen"],
                    "classifier__shrinkage": [
                        "auto", 0.1, 0.3, 0.5, 0.7
                    ],
                },
            ],

            "quadratic_discriminant_analysis": {
                "classifier__reg_param": [
                    0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8
                ],
                "classifier__tol": [
                    1e-4, 1e-3, 1e-2
                ],
            },

            "bayesian_discrete": {
                "classifier__alpha": [
                    0.001, 0.01, 0.1, 0.5,
                    1.0, 2.0, 5.0, 10.0
                ],
                "classifier__fit_prior": [True, False],
                "classifier__min_categories": [None, 2, 5],
            },

            "bayesian_continuous": {
                "classifier__var_smoothing": [
                    1e-12, 1e-11, 1e-10, 1e-9,
                    1e-8, 1e-7, 1e-6, 1e-5
                ],
            },

            "artificial_neural_network": {
                "classifier__hidden_layer_sizes": [
                    (64,),      # (50,),
                    (128,),     # (100,),
                    (256,),     # (150,),
                    (512,),     # (200,),
                    (128, 128), # (50, 50),
                    (256, 218), # (100, 50),
                    (256, 256), # (100, 100),
                    (128, 64)   # (128, 64),
                ],
                "classifier__activation": ["relu", "tanh"],
                "classifier__solver": ["adam"],
                "classifier__alpha": [0.0001, 0.001, 0.01, 0.1],
                "classifier__learning_rate_init": [0.0001, 0.001, 0.01],
                "classifier__batch_size": [32, 64, 128, "auto"],
            }
        }

    def tune_model(
        self,
        model_name,
        X_train,
        y_train,
        scoring = 'f1_macro',
        n_iter  = 30,
        cv      = 5,
    ):
        """Tune a classifier using stratified cross-validation."""

        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        search = RandomizedSearchCV(
            estimator           = self.models[model_name],
            param_distributions = self.param_grids[model_name],
            n_iter              = n_iter,
            scoring             = scoring,
            cv = StratifiedKFold(
                n_splits     = cv,
                shuffle      = True,
                random_state = self.random_state,
            ),
            random_state = self.random_state,
            n_jobs       = -1,
            verbose      = 1,
            refit        = True,
            error_score  = "raise"
        )

        search.fit(X_train, y_train)

        # Replace original estimator with optimized estimator
        self.models[model_name] = search.best_estimator_

        return {
            'model'          : model_name,
            'best_score'     : search.best_score_,
            'best_params'    : search.best_params_,
            'best_estimator' : search.best_estimator_,
        }
    
    @staticmethod
    def _validate_discrete_features(X, feature_names=None) -> None:

        values = X.data if issparse(X) else np.asarray(X).ravel()

        non_integer = ~np.isclose(values, np.round(values))
        negative = values < 0

        if np.any(non_integer):
            print("Non-integer values:", values[non_integer][:10])

        if np.any(negative):
            print("Negative values:", values[negative][:10])

        if np.any(non_integer) or np.any(negative):
            raise ValueError(
                "CategoricalNB requires non-negative integer feature codes."
            )

    def get_models(self) -> dict[str, BaseEstimator]:
        """Return the available, unfitted estimators."""
        return self.models.copy()

    def fit(
        self,
        X_train     : np.ndarray | None = None,
        y_train     : np.ndarray | None = None,
        model_names : list[str] | None = None,
    ) -> dict[str, BaseEstimator]:

        X_train = (
            self.X if X_train is None
            else X_train if issparse(X_train)
            else np.asarray(X_train)
        )

        y_train = (
            self.y if y_train is None
            else np.asarray(y_train).ravel()
        )

        if X_train.ndim != 2:
            raise ValueError(
                f"X_train must be 2-D; got shape {X_train.shape}."
            )

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train have different sample counts: "
                f"{X_train.shape[0]} vs {y_train.shape[0]}."
            )

        selected = (
            self.models if model_names is None
            else {name: self.models[name] for name in model_names}
        )

        self.fitted_models = {}

        # Estimators that require dense input
        dense_models = {
            "linear_discriminant_analysis",
            "quadratic_discriminant_analysis",
            "bayesian_continuous",
        }

        for name, estimator in selected.items():
            X = X_train
            y = y_train
            if name == "bayesian_discrete":
                self._validate_discrete_features(X)

            if name in dense_models and issparse(X):
                    X = X.toarray()
                
            estimator.fit(X, y_train)

            self.fitted_models[name] = estimator

        return self.fitted_models

    def predict(
        self,
        X          : np.ndarray,
        model_name : str,
    ) -> np.ndarray:
        """Predict labels using a fitted model."""
        if model_name not in self.fitted_models:
            raise ValueError(f"Model {model_name!r} has not been fitted.")
        return self.fitted_models[model_name].predict(np.asarray(X))

    def fit_evaluate(
        self,
        test_size: float = 0.2,
        random_state: int | None = None,
        model_names: list[str] | None = None,
        stratify: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """
        Split the dataset, fit each selected model on the training set, and
        evaluate it on the held-out test set.

        Returns a dictionary containing metrics, predictions, and fitted
        estimators for each model.
        """

        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        seed = self.random_state if random_state is None else random_state
        stratify_labels = self.y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_labels,
        )

        self.fit(X_train, y_train, model_names=model_names)

        results: dict[str, dict[str, Any]] = {}

        for name, estimator in self.fitted_models.items():

            y_pred = estimator.predict(X_test)

            results[name] = {
                # Evaluation metrics
                "accuracy": accuracy_score(y_test, y_pred),

                "precision_macro": precision_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),

                "recall_macro": recall_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),

                "f1_macro": f1_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),

                # Detailed evaluation
                "confusion_matrix": confusion_matrix(
                    y_test,
                    y_pred,
                ),

                "classification_report": classification_report(
                    y_test,
                    y_pred,
                    output_dict=True,
                    zero_division=0,
                ),

                # Predictions and fitted estimator
                "y_test": y_test,
                "y_pred": y_pred,
                "estimator": estimator,
            }

        return results
    
    def get_rule_induction_rules(self) -> str:

        if "rule_induction" not in self.fitted_models:
            raise ValueError("Fit the rule_induction model first.")

        pipeline = self.fitted_models["rule_induction"]
        ripper = pipeline.named_steps["classifier"].estimator_

        return ripper.out_model()

    def summary(self) -> str:
        """Return a compact description of the available classifiers."""
        descriptions = {
            "knn": "K-Nearest Neighbors",
            "classification_tree": "Decision Tree Classifier",
            "rule_induction": "Interpretable tree-based rule induction baseline",
            "logistic_regression": "Logistic Regression",
            "linear_discriminant_analysis": "Linear Discriminant Analysis",
            "quadratic_discriminant_analysis": "Quadratic Discriminant Analysis",
            "bayesian_discrete": "Categorical Naive Bayes",
            "bayesian_continuous": "Gaussian Naive Bayes",
            "svm": "Support Vector Machine (RBF kernel)",
            "artificial_neural_network": "Multi-Layer Perceptron",
        }
        return "\n".join(f"{name}: {description}" for name, description in descriptions.items())