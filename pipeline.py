from components.data_processor import (
    _impl_preprocess_census_data_,
    _impl_save_processed_data_,
    _impl_load_processed_data_,
    _impl_if_processed_data_exist_
)

from components.feature_selector import (
    auto_features_selector, 
    semantic_features_selector, 
    LayeredFeatureSelector
)

from components.base_classifier  import BaseClassifier
from components.meta_classifier  import MetaClassifier

from pathlib         import Path
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

DATA_SOURCE               = './data/census.csv'
PROCESSED_DATA_OUTPUT_DIR = './data/processed' + '-15000'

RESULTS_DIR = './results'


def process_source_data(
    sample_percentage: float = 0.1
):
    global DATA_SOURCE

    df = pd.read_csv(DATA_SOURCE)

    # Randomly sample a percentage of the entire dataset
    if not 0 < sample_percentage <= 1.0:
        raise ValueError("sample_percentage must be between 0 and 100.")

    df = df.sample(
        frac = sample_percentage,
        random_state = 42
    ).reset_index(drop = True)

    # Preprocess (do not fit or transform yet)
    X, y, preprocessor = _impl_preprocess_census_data_(df)

    return X, y, preprocessor


def save_processed_data(
    X,
    y,
    preprocessor
):
    _impl_save_processed_data_(
        X,
        y,
        preprocessor,
        output_dir=PROCESSED_DATA_OUTPUT_DIR
    )


def load_processed_data():
    global PROCESSED_DATA_OUTPUT_DIR
    # Inspect training dataset
    X_train, X_test, y_train, y_test, feature_names = (
        _impl_load_processed_data_(
            data_dir    = PROCESSED_DATA_OUTPUT_DIR,
            train_ratio = 0.8,
            show_report = False
        )
    )

    return X_train, y_train, X_test, y_test, feature_names


def processed_data_exist():
    global PROCESSED_DATA_OUTPUT_DIR

    output_dir = Path(PROCESSED_DATA_OUTPUT_DIR).resolve()
    
    return _impl_if_processed_data_exist_(output_dir)


def export_results(
    results,
    file_name,
    show_report=False
):
    # Display and collect results
    results_data = []

    for model_name, metrics in results.items():
        if show_report:
            print(f"\n{model_name}")
            print(f"  Accuracy:        {metrics['accuracy']:.4f}")
            print(f"  Precision macro: {metrics['precision_macro']:.4f}")
            print(f"  Recall macro:    {metrics['recall_macro']:.4f}")
            print(f"  F1 macro:        {metrics['f1_macro']:.4f}")

            print("\n", classification_report(
                metrics["y_test"],
                metrics["y_pred"],
                zero_division=0
            ))

        row = {
            "model"           : model_name,
            "accuracy"        : metrics["accuracy"],
            "precision_macro" : metrics["precision_macro"],
            "recall_macro"    : metrics["recall_macro"],
            "f1_macro"        : metrics["f1_macro"],
        }

        # Flatten classification report
        report = metrics['classification_report']

        if isinstance(report, str):
            from sklearn.metrics import classification_report as get_report

            # If only the text report is available, it cannot be reliably
            # converted back to structured metrics in every case.
            raise ValueError(
                "classification_report must be a dictionary. "
                "Generate it using output_dict=True."
            )

        for label, scores in report.items():
            if isinstance(scores, dict):
                for metric_name, value in scores.items():
                    row[f"{label}_{metric_name}"] = value
            else:
                row[f"{label}_{'value'}"] = scores

        results_data.append(row)

    # Export flattened results
    results_df = pd.DataFrame(results_data)

    output_path = Path(os.path.join(RESULTS_DIR, file_name))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nResults exported to: {output_path.resolve()}")


def select_features(
    X_train,
    y_train,
    method_name    = 'kbest',
    features       = None,
    k              = 0,
    semantic_names = None,
):
    """
    Optional semantic filtering followed by automatic feature selection.
    """

    # Stage 1: Semantic feature filtering
    X_semantic, semantic_features, semantic_indices = (
        semantic_features_selector(
            X_train=X_train,
            semantic_names=semantic_names,
            semantic_csv=os.path.join(
                PROCESSED_DATA_OUTPUT_DIR,
                "semantic_feature_groups.csv"
            ),
            feature_names=features,
        )
    )

    # Stage 2: Automatic feature selection
    X_train_selected, selected_features, selector = (
        auto_features_selector(
            X_train       = X_semantic,
            y_train       = y_train,
            feature_names = semantic_features,
            method        = method_name,
            k             = k,
        )
    )
    
    print("\n========== Feature Selection Summary ==========")

    print(f"Semantic filter: {semantic_names if semantic_names else 'None'}")
    print(f"Original features: {X_train.shape[1]}")
    print(f"Semantic candidate features: {len(semantic_indices)}")
    print(f"Selection method: {method_name}")
    print(f"Selected features: {len(selected_features)}")

    print("===============================================")

    layered_selector = LayeredFeatureSelector(
        semantic_indices=semantic_indices,
        feature_selector=selector
    )

    return X_train_selected, selected_features, layered_selector




def fine_tune_model(
    classifier,
    model_name : str
):
    result = classifier.tune_model(
            model_name = model_name,
            X_train    = classifier.X,
            y_train    = classifier.y,
            scoring    = 'f1_macro',
            n_iter     = 30,
            cv         = 5,
        )

    return result

def evaluate_model(classifier):
    results = classifier.fit_evaluate(test_size=0.2)
    export_results(results, 'results_logistic.csv', show_report=True)


NUMERIC_SEMANTIC_FEATURES = [
    'age', 
    'wage per hour',
    'capital gains',
    'capital losses',
    'dividends from stocks',
    'num persons worked for employer',
    'own business or self employed',
    'veterans benefits',
    'weeks worked in year'
]

CATEGORICAL_SEMANTIC_FEATURES = [
    'class of worker',
    'detailed industry recode',
    'detailed occupation recode',
    'education',
    'enroll in edu inst last wk',
    'marital stat',
    'major industry code',
    'major occupation code',
    'race',
    'hispanic origin',
    'sex',
    'member of a labor union',
    'reason for unemployment',
    'full or part time employment stat',
    'tax filer stat',
    'region of previous residence',
    'state of previous residence',
    'detailed household and family stat',
    'detailed household summary in household',
    'migration code-change in msa',
    'migration code-change in reg',
    'migration code-move within reg',
    'live in this house 1 year ago',
    'migration prev res in sunbelt',
    'family members under 18',
    'country of birth father',
    'country of birth mother',
    'country of birth self',
    'citizenship',
    "fill inc questionnaire for veteran's admin",
    'year'
]


if __name__ == '__main__':

    
    #### Available Base Classifier
    ## Base Classifier
    ## knn
    ## rule_induction
    ## svm
    ## classification_tree
    ## logistic_regression
    ## Bayesian_continuous
    ## Bayesian_discrete
    ## linear_discriminant_analysis
    ## quadratic_discriminant_analysis
    ## artificial_networks

    classifier_name = 'logistic_regression'    
    dataset_size    = 0.01
    feature_method  = 'kbest'    
    feature_number  = 30

    ''' Load and Split Data '''

    if not processed_data_exist():
        print("Processed Data Not Exist. Start Transform...")
        X, y, preprocessor = process_source_data(sample_percentage=dataset_size)
        save_processed_data(X, y, preprocessor)
        

    X_train, y_train, X_test, y_test, feature_names = load_processed_data()
    
    X_train_selected, selected_features, selector = select_features(
        X_train        = X_train, 
        y_train        = y_train,
        method_name    = feature_method,
        features       = feature_names,
        k              = feature_number,
        # semantic_names = CATEGORICAL_SEMANTIC_FEATURES
    )
    
    # Apply the same selector to test data
    if selector is not None:
        X_test_selected = selector.transform(X_test)
    else:
        X_test_selected = X_test

    print("Selected Features:", selected_features)
    # print("Selector:", selector)
    print("Training shape:", X_train_selected.shape)
    print("Test shape:", X_test_selected.shape)


    ''' Base Classification '''

    # classifier = BaseClassifier(X = X_train_selected, y = y_train)

    # evaluate_model(classifier)

    # evaluate_model(classifier)
    # print("Target distribution:")
    # print(pd.Series(y_train).value_counts())

    # print("\nTarget proportions:")
    # print(pd.Series(y_train).value_counts(normalize=True))

    # print("\nUnique labels:", np.unique(y_train))
    
    # result = fine_tune_model(classifier, classifier_name)
    # print("Best score:", result["best_score"])
    # print("Best parameters:", result["best_params"])

    # with open(f'./evaluation/finetune/{classifier_name}/{feature_method}-{feature_number}-{dataset_size}.txt', "a") as f:
    #     print("Training shape:", X_train_selected.shape, file=f)
    #     print("Test shape:", X_test_selected.shape, file=f)
    # 
    #     print("Best score:", result["best_score"], file=f)
    #     print("Best parameters:", result["best_params"], file=f)

    
    '''Meta Classifier'''

    '''
    meta_classifier = MetaClassifier(
        base_classifier=classifier,
        model_names=[
            # 'knn',
            'svm',
            'classification_tree',
            'logistic_regression'
        ],
        final_estimator='logistic_regression',
        cv=5,
        stack_method='auto',
        passthrough=False
    )

    # Fine-tune meta-classifier
    meta_result = meta_classifier.tune_model(
        X_train_selected,
        y_train,
        scoring = 'f1_macro',
        n_iter  = 20,
        cv      = 5
    )

    meta_result_eval = meta_classifier.evaluate(
        X_test_selected,
        y_test
    )

    print("\nStacking Classifier")
    print("Best CV Score:", meta_result['best_score'])
    print("Best Parameters:", meta_result['best_params'])
    print("Test Macro-F1:", meta_result_eval['f1_macro'])
    print(f"  Accuracy:        {meta_result_eval['accuracy']:.4f}")
    print(f"  Precision macro: {meta_result_eval['precision_macro']:.4f}")
    print(f"  Recall macro:    {meta_result_eval['recall_macro']:.4f}")
    print(f"  F1 macro:        {meta_result_eval['f1_macro']:.4f}")

    print("\nClassification Report")
    print(
        classification_report(
            meta_result_eval["y_test"],
            meta_result_eval["y_pred"],
            digits=2,
            zero_division=0
        )
    )

    print("\nConfusion Matrix")
    print(meta_result_eval["confusion_matrix"])
    '''
    