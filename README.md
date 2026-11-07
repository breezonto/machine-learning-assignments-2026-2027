# MACHINE LERANING UPM MUCD

Note: The codebase is currently somewhat unstructured due to rapid experimentation and ongoing development. Experimental results are also distributed across multiple log files. In the coming weeks, the codebase will be progressively refactored to improve modularity, integration and overall organization. The project remains under active development.


## Project Structure

```
/
├── data/
|   ├── categorical-encoding.yml
|   ├── census.csv
|   └── import-census-data.sql
|
├── components/ # components of designed classifier architecture
|   ├── base_classifier.py
|   ├── data_processor.py
|   ├── evaluator.py
|   ├── meta_classifier.py
|   └── visualizer.py
|
├── pipeline.py   # pipeline integrate each component
├── vis.py/viz.py # for data visualiztion, it will be migrated into components/visualizer.py in the future
├── README.md

```

## Models & Algorithms

According to the lecture presentations, this report investigates 12 models for supervised classification tasks.

| Approach | Representative algorithm | Main experimental dimension |
|---|---|---|
1 | KNN	| KNeighborsClassifier | Distance metrics and neighborhood size |
2 | Rule Induction | RIPPER | Rule complexity and interpretability |
3 | Artificial Networks | MLP | Network architecture and representation |
4 | SVM | RBF SVM | Kernel selection and regularization |
5 | Classification Trees | CART | Tree depth and pruning |
6 | Feature Subset Selection | RFE / KBest / CFS | Feature relevance and dimensionality |
7 | Logistic Regression | LogisticRegression | Regularization and linear decision boundaries |
8 | Discriminant Analysis | LDA / QDA | Covariance estimation and distributional assumptions |
9 | Bayesian (Discrete) | Categorical Naive Bayes | Discretization and categorical likelihoods |
10 | Bayesian (Continuous) | Gaussian Naive Bayes | Continuous distribution assumptions |
11 | Metaclassifiers | Logistic Regression / Stacking | Ensemble strategies |

## Dataset Selection: Census-Income (KDD) From 1994 To 1995

The Census-Income (KDD) dataset is a publicly available dataset from the U.S. Census Bureau, distributed through the UCI Machine Learning Repository. It contains demographic, employment, educational, financial, household, and migration-related information collected from the 1994 and 1995 U.S. Current Population Survey (CPS). The dataset comprises approximately 299,000 instances and 42 attributes, including 41 input features and one binary target variable, income, indicating whether an individual's annual income is below $50,000 or at least $50,000. It also includes a sampling weight (MARSUPWT) to account for the population represented by each observation. Due to its combination of heterogeneous categorical and numerical features, substantial class imbalance, and relatively large sample size, the dataset provides a useful benchmark for evaluating classification algorithms and data preprocessing techniques. The source dataset URL please refers: [Census-Income \(KDD\)](https://archive.ics.uci.edu/dataset/117/census%2Bincome%2Bkdd)