"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, make_scorer
)
from sklearn.metrics import f1_score


def split_dataset(df, preprocessor):
    target_variable = 'Attrition'
    y = df[target_variable]
    x = df.drop(columns=[target_variable])

    strat_shuf_split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(strat_shuf_split.split(x, y))
    x_train = x.iloc[train_idx, :]
    y_train = y[train_idx]
    x_test = x.loc[test_idx, :]
    y_test = y[test_idx]

    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)

    return x_train, y_train, x_test, y_test


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    report = {}
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro')
    }
    skf = StratifiedKFold(shuffle=True, random_state=42, n_splits=3)

    for i in range(len(list(models))):
        model = list(models.values())[i]
        para=param[list(models.keys())[i]]

        gs = GridSearchCV(model, para, cv=skf, scoring=scoring, refit='f1')
        gs.fit(X_train,y_train)

        model.set_params(**gs.best_params_)
        model.fit(X_train,y_train)

        # y_train_pred = model.predict(X_train)

        y_test_pred = model.predict(X_test)


        report[list(models.keys())[i]] = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred)
        }

    return report


def train_model(x_train, y_train, x_test, y_test):
    
    models = {
        "Logistic Regression": LogisticRegression(),
        "KNeighbors Classifier": KNeighborsClassifier(),
        "Support Vector Machine": SVC(),
        "Random Forest": RandomForestClassifier(),
        "GradientBoosting Classifier": GradientBoostingClassifier(),
        "AdaBoost Classifier": AdaBoostClassifier(),
        "XGB Classifier": XGBClassifier()
    }
    params={
        "Logistic Regression": {
            'penalty':['l2', 'l1'],
            'solver':['liblinear']
        },
        "KNeighbors Classifier":{
            'n_neighbors':[5, 7],
            'weights': ['uniform', 'distance']
        },
        "Support Vector Machine":{
            'kernel':['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        "Random Forest":{
            'n_estimators': [100, 200]
        },
        "GradientBoosting Classifier":{
            'n_estimators': [100, 200]
        },
        "AdaBoost Classifier":{
            'n_estimators': [100, 200]
        },
        "XGB Classifier":{
            'n_estimators': [100, 200]
        }
        
    }

    model_report = evaluate_models(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test,
                                        models=models, param=params)
    
    # ## To get best model score from dict
    best_model_name, _ = sorted([(model, score) for model, scores in model_report.items() for metric, score in scores.items() if metric == 'f1'], reverse=True, key= lambda x: x[1])[0]

    best_model = models[best_model_name]
    # return model_report, best_model_name, best_model
    return best_model


# def evaluate_model(
#     model: Any,
#     x_test: pd.DataFrame,
#     y_test: pd.Series
# ):
#     score, params = model.best_score_, model.best_params_
#     print("Best score: ", score)
#     print("Best params: ", params)
#     predictions = model.predict(x_test)
#     print(classification_report(y_test, predictions))
#     print(model.cv_results_['mean_test_f1'])
#     cr = classification_report(y_test, predictions, output_dict=True)
#     df_cr = pd.DataFrame(cr).iloc[:-1, :].T
#     sns.heatmap(df_cr, annot=True)

#     return plt
