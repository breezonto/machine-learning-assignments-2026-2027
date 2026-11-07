from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
    SelectFromModel,
    SelectPercentile,
)

from sklearn.ensemble          import RandomForestClassifier
from sklearn.feature_selection import SelectorMixin
from sklearn.base              import BaseEstimator, TransformerMixin
from sklearn.preprocessing     import KBinsDiscretizer
from sklearn.metrics           import mutual_info_score

from scipy import sparse

import pandas as pd
import numpy  as np


class CFSSelector:
    """Correlation-Based Feature Selection (Hall, 1999).

    Uses symmetrical uncertainty to measure feature relevance
    and redundancy, with greedy forward feature selection.
    """

    def __init__(self, k=10, n_bins=10):
        self.k = k
        self.n_bins = n_bins
        self.support_ = None

    @staticmethod
    def _to_dense_array(X):
        """Convert sparse matrices or other inputs into a dense NumPy array."""
        if sparse.issparse(X):
            X = X.toarray()

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                f"Expected a 2D feature matrix, got shape {X.shape}."
            )

        return X

    @staticmethod
    def _entropy(x):
        """Calculate Shannon entropy."""
        _, counts = np.unique(x, return_counts=True)
        probabilities = counts / counts.sum()

        return -np.sum(
            probabilities * np.log2(probabilities)
        )

    @classmethod
    def _symmetrical_uncertainty(cls, x, y):
        """Calculate symmetrical uncertainty."""
        hx = cls._entropy(x)
        hy = cls._entropy(y)

        if hx + hy == 0:
            return 0.0

        mi = mutual_info_score(x, y)

        return 2.0 * mi / ((hx + hy) * np.log(2))

    def fit(self, X, y):
        """Fit the CFS selector using greedy forward selection."""

        X = self._to_dense_array(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("Expected a 1D target array.")

        n_samples, n_features = X.shape

        if len(y) != n_samples:
            raise ValueError(
                "X and y contain different numbers of samples."
            )

        if not 1 <= self.k <= n_features:
            raise ValueError("Invalid number of selected features.")

        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2.")

        # 1. Discretize continuous features.
        X_discrete = X.copy()

        for i in range(n_features):
            unique_values = np.unique(X[:, i])

            if len(unique_values) > 20:
                discretizer = KBinsDiscretizer(
                    n_bins=min(self.n_bins, len(unique_values)),
                    encode="ordinal",
                    strategy="quantile"
                )

                X_discrete[:, i] = discretizer.fit_transform(
                    X[:, i].reshape(-1, 1)
                ).ravel()

        # 2. Calculate feature-to-class relevance.
        relevance = np.array([
            self._symmetrical_uncertainty(
                X_discrete[:, i], y
            )
            for i in range(n_features)
        ])

        # 3. Cache feature-to-feature redundancy.
        redundancy_cache = {}

        def get_redundancy(i, j):
            key = (min(i, j), max(i, j))

            if key not in redundancy_cache:
                redundancy_cache[key] = (
                    self._symmetrical_uncertainty(
                        X_discrete[:, i],
                        X_discrete[:, j]
                    )
                )

            return redundancy_cache[key]

        # 4. Greedy forward selection.
        selected = []
        remaining = set(range(n_features))

        while len(selected) < self.k:
            best_feature = None
            best_merit = -np.inf

            for candidate in remaining:
                candidate_features = selected + [candidate]
                subset_size = len(candidate_features)

                relevance_sum = np.sum(
                    relevance[candidate_features]
                )

                redundancy_sum = sum(
                    get_redundancy(
                        candidate_features[i],
                        candidate_features[j]
                    )
                    for i in range(subset_size)
                    for j in range(i + 1, subset_size)
                )

                # CFS subset merit.
                if subset_size == 1:
                    merit = relevance_sum

                else:
                    average_redundancy = (
                        2 * redundancy_sum /
                        (subset_size * (subset_size - 1))
                    )

                    merit = (
                        relevance_sum /
                        np.sqrt(
                            subset_size +
                            subset_size * (subset_size - 1)
                            * average_redundancy
                        )
                    )

                if merit > best_merit:
                    best_merit = merit
                    best_feature = candidate

            selected.append(best_feature)
            remaining.remove(best_feature)

        # 5. Store selected feature mask.
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[selected] = True

        return self

    def transform(self, X):
        """Select features using the fitted support mask."""

        if self.support_ is None:
            raise RuntimeError(
                "The selector must be fitted before calling transform()."
            )

        X = self._to_dense_array(X)

        if X.shape[1] != len(self.support_):
            raise ValueError(
                "Input feature count differs from the fitted data."
            )

        return X[:, self.support_]

    def fit_transform(self, X, y):
        """Fit the selector and transform the input data."""

        return self.fit(X, y).transform(X)

    def get_support(self):
        """Return the selected feature mask."""

        if self.support_ is None:
            raise RuntimeError(
                "The selector must be fitted before calling get_support()."
            )

        return self.support_


def auto_features_selector(
    X_train,
    y_train,
    feature_names,
    method = "kbest",
    k=0
):
    """Perform feature selection on training data."""

    n_features = X_train.shape[1]

    if len(feature_names) != n_features:
        raise ValueError(
            "Feature names do not match the input matrix dimensions."
        )

    if not 0 <= k <= n_features:
        raise ValueError(
            f"k must be between 0 and {n_features}."
        )

    # 1. Skip feature selection when k = 0
    if k == 0:
        # print("\n========== Feature Selection ==========")
        # print("Method: None (all features retained)")
        # print(f"Original features: {n_features}")
        # print(f"Selected features: {n_features}")

        return X_train, list(feature_names), None

    # 2. Initialize feature selector
    if method == "variance":
        selector = VarianceThreshold(threshold=0.01)

    elif method == "kbest":
        selector = SelectKBest(
            score_func=f_classif,
            k=k
        )

    elif method == "cfs":
        selector = CFSSelector(k=k)
    
    elif method == "rfe":
        selector = RFE(
            estimator=RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            n_features_to_select=k
        )

    elif method == "model":
        selector = SelectFromModel(
            estimator=RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            threshold="median"
        )

    else:
        raise ValueError(
            f"Unknown feature selection method: {method}"
        )

    # 3. Fit selector and transform training data
    X_selected = selector.fit_transform(X_train, y_train)

    # 4. Retrieve selected feature names
    support = selector.get_support()

    selected_features = [
        name
        for name, selected in zip(feature_names, support)
        if selected
    ]

    # print("\n========== Feature Selection ==========")
    # print(f"Method: {method}")
    # print(f"Original features: {n_features}")
    # print(f"Selected features: {len(selected_features)}")

    return X_selected, selected_features, selector


class SemanticFeatureSelector(SelectorMixin):
    """Select columns based on semantic feature groups."""

    def __init__(self, semantic_names, semantic_features):
        self.semantic_names = semantic_names
        self.semantic_features = semantic_features

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]

        selected_indices = []

        for name in self.semantic_names:
            row = self.semantic_features[
                self.semantic_features["semantic_feature"] == name
            ]

            if row.empty:
                raise ValueError(f"Unknown semantic feature: {name}")

            indices = row.iloc[0]["indices"]

            if isinstance(indices, str):
                indices = [int(i) for i in indices.split(",")]
            else:
                indices = [int(indices)]

            selected_indices.extend(indices)

        self.selected_indices_ = np.array(
            sorted(set(selected_indices)), dtype=int
        )

        if np.any(self.selected_indices_ >= self.n_features_in_):
            raise ValueError("Semantic feature indices exceed X dimensions.")

        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[self.selected_indices_] = True

        return self

    def _get_support_mask(self):
        return self.support_


def semantic_features_selector(
    X_train,
    semantic_names: str | list[str] | None = None,
    semantic_csv=None,
    feature_names=None,
):
    """
    Filter candidate features using semantic feature groups.

    If semantic_names is None, retain all features.
    """

    if feature_names is None:
        raise ValueError("feature_names must be provided.")

    if len(feature_names) != X_train.shape[1]:
        raise ValueError(
            "Feature names do not match input matrix dimensions."
        )

    # Default: retain all features
    if semantic_names is None:
        return X_train, list(feature_names), np.arange(X_train.shape[1])

    if isinstance(semantic_names, str):
        semantic_names = [semantic_names]

    semantic_df = pd.read_csv(semantic_csv)

    selected_indices = []

    for name in semantic_names:
        row = semantic_df[
            semantic_df["semantic_feature"] == name
        ]

        if row.empty:
            raise ValueError(f"Unknown semantic feature: {name}")

        indices = [
            int(i)
            for i in str(row.iloc[0]["indices"]).split(",")
        ]

        selected_indices.extend(indices)

    selected_indices = np.array(sorted(set(selected_indices)))

    X_selected = X_train[:, selected_indices]

    selected_features = np.asarray(feature_names)[
        selected_indices
    ].tolist()

    return X_selected, selected_features, selected_indices


class LayeredFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, semantic_indices, feature_selector):
        self.semantic_indices = semantic_indices
        self.feature_selector = feature_selector

    def transform(self, X):
        X_semantic = X[:, self.semantic_indices]

        if self.feature_selector is not None:
            return self.feature_selector.transform(X_semantic)

        return X_semantic
