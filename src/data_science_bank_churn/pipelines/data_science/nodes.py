"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, make_scorer
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def split_data(df: pd.DataFrame, target_variable: str, model_options_lg: Dict) -> Tuple:
    y = df[target_variable]
    x = df.drop(columns=[target_variable])

    strat_shuf_split = StratifiedShuffleSplit(n_splits=1,
                                              test_size=model_options_lg['test_size'],
                                              random_state=model_options_lg['random_state'])

    train_idx, test_idx = next(strat_shuf_split.split(x, y))
    x_train = df.loc[train_idx, x.columns]
    y_train = df.loc[train_idx, target_variable]
    x_test = df.loc[test_idx, x.columns]
    y_test = df.loc[test_idx, target_variable]

    return x_train, y_train, x_test, y_test


def train_model(x_train: pd.DataFrame, y_train: pd.Series, model_options_lg: Dict) -> Any:
    skf = StratifiedKFold(shuffle=True,
                          random_state=model_options_lg['random_state'],
                          n_splits=model_options_lg['n_splits'])

    ss = StandardScaler()

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro')
    }

    if model_options_lg['model'] == "LogisticRegression":
        lreg = LogisticRegression()

        estimator = Pipeline([
            # ("polynomial_features", PolynomialFeatures()),
            ("scaler", ss),
            ("logistic_regression", lreg)])

        params = {
            # 'polynomial_features__degree': [1, 2, 3],
            'logistic_regression__penalty': ['l1', 'l2'],
            'logistic_regression__C': [4, 6, 10],
            'logistic_regression__solver': ['liblinear']
        }

    elif model_options_lg['model'] == "SVC":
        svc = SVC()

        estimator = Pipeline([
            # ("polynomial_features", PolynomialFeatures()),
            ("scaler", ss),
            ("svc_classifier", svc)])

        params = {
            # 'polynomial_features__degree': [1, 2,3],
            'svc_classifier__C': [2, 4, 6],
            'svc_classifier__kernel': ['rbf', 'sigmoid']
        }
    elif model_options_lg['model'] == "RandomForest":
        rf = RandomForestClassifier()

        estimator = Pipeline([
            # ("polynomial_features", PolynomialFeatures()),
            ("scaler", ss),
            ("RF_classifier", rf)])

        params = {
            # 'polynomial_features__degree': [1, 2,3],
            'RF_classifier__n_estimators': [350, 400, 450],
            'RF_classifier__max_depth': [None, 20],
            'RF_classifier__warm_start': [True]
        }

    grid = GridSearchCV(estimator, params, scoring=scoring, refit='f1', cv=skf, n_jobs=-1)
    grid.fit(x_train, y_train)

    return grid


def evaluate_model(model: Any, x_test: pd.DataFrame, y_test: pd.Series):
    score, params = model.best_score_, model.best_params_
    print("Best score: ", score)
    print("Best params: ", params)
    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions))
    print(model.cv_results_['mean_test_f1'])
    cr = classification_report(y_test, predictions, output_dict=True)
    df_cr = pd.DataFrame(cr).iloc[:-1, :].T
    sns.heatmap(df_cr, annot=True)

    return plt
