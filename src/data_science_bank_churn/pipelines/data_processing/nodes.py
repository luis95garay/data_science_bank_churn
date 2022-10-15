"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.stats.mstats import normaltest
from scipy.stats import boxcox
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import f_classif

def remove_unnecessary_columns(df: pd.DataFrame, delete_columns: Dict) -> pd.DataFrame:
    df.drop(columns=delete_columns['selected_columns'], inplace=True)

    missing_th = int((1 - delete_columns['threshold']) * len(df)) + 1

    missing_data_cols = [col for col in df.columns.tolist() if df[col].count() < missing_th]
    logger = logging.getLogger(__name__)
    if len(missing_data_cols) > 0:
        df.drop(columns=missing_data_cols, inplace=True)
        logger.info("Incomplete deleted columns: ", missing_data_cols)
    else:
        logger.info("There are not deleted columns")

    return df


# def remove_incomplete_rows(df: pd.DataFrame)-> pd.DataFrame:
#     missing_data_cols = [col for col in df.columns.tolist() if len(df[pd.isnull(df[col]) == True]) > 0]
#     for column in missing_data_cols:
#         null_values = df[pd.isnull(df[column]) == True].index.tolist()
#         df.drop(null_values,axis=0, inplace=True)
#         #print(column, '\t', len(null_values), '\t', df[column].dtypes)
#     print("Incomplete deleted rows: ", missing_data_cols)
#     return df

def reduce_categorical_column_options(df: pd.DataFrame, reduce_options_columns: Dict) -> pd.DataFrame:
    df_data_red = df.copy()
    mask = df.dtypes == object
    categorical_cols = df.columns[mask]
    combined_col_names = []
    for col in categorical_cols:
        if col not in reduce_options_columns['exclude']:
            val_counts = df[col].value_counts()
            replace_cats = list(val_counts[(val_counts / val_counts.sum()) < reduce_options_columns['threshold']].index)
            if len(replace_cats) > 0:
                df_data_red[col] = df_data_red.replace(replace_cats, 'others')[col]
                combined_col_names.append(col)

    if len(combined_col_names) > 0:
        print("Reduced columns: ", combined_col_names)
    else:
        print("Not reduced columns")

    return df_data_red


def handle_outliers(df: pd.DataFrame, outliers_columns: Dict) -> pd.DataFrame:
    # mask = df.dtypes == float
    # numerical_cols = df.columns[mask]
    #numerical_cols = ['Year', 'Month', 'Children']
    print("Shape before removing: ", df.shape)
    transformed_columns = []

    for col in outliers_columns:
        p_value = normaltest(df[col].values)[1]
        if p_value < 0.05:
            uppper_boundary = df[col].mean() + 3 * df[col].std()
            lower_boundary = df[col].mean() - 3 * df[col].std()
        else:
            IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
            lower_boundary = df[col].quantile(0.25) - (IQR * 1.5)
            uppper_boundary = df[col].quantile(0.75) + (IQR * 1.5)
        outliers = df[(df[col] < lower_boundary) | (df[col] > uppper_boundary)].index.tolist()

        if len(outliers) > 0:
            df.drop(outliers, axis=0, inplace=True)
            transformed_columns.append((col, len(outliers)))

    if len(transformed_columns) > 0:
        print("Outliers deleted: ", transformed_columns)
        print("Shape after removing: ", df.shape)
    else:
        print("There are not outliers")
    return df


def treat_skewed_columns(df: pd.DataFrame, skewed_columns: Dict) -> pd.DataFrame:
    mask_float = df.dtypes == np.float64
    float_cols = df.columns[mask_float].tolist()
    mask_int = df.dtypes == np.int64
    int_cols = df.columns[mask_int].tolist()
    numerical_cols = float_cols + int_cols

    if len(numerical_cols) > 0 and len(skewed_columns['exclude_columns']) > 0:
        for column in skewed_columns['exclude_columns']:
            numerical_cols.remove(column)

    transformed_columns = []
    if skewed_columns['method'] == "boxcox":
        boxcox_dict = {}
        for col in numerical_cols:
            print(col, min(df[col]))
            df[col].fillna(0, inplace=True)
            boxcox_current, lam = boxcox(df[col])
            boxcox_dict.update({f"{col}": [boxcox_current, lam]})
            df[col] = boxcox_current
    elif skewed_columns['method'] == "log":
        for col in numerical_cols:
            p_value = normaltest(df[col].values)[1]
            if p_value > 0.05:
                #print(col, p_value)
                if df[col].min() >= 0:
                    df[col] = (df[col] + 1).transform(np.log)
                else:
                    df[col] = (df[col] - df[col].min() + 1).transform(np.log)
                transformed_columns.append(col)

    if len(transformed_columns) > 0:
        print("Transformed columns: ", transformed_columns)
    else:
        print("There are not transformed columns")

    return df


def encode_categorical_columns(df: pd.DataFrame, target_variable: str) -> pd.DataFrame:
    lb, le, ore = LabelBinarizer(), LabelEncoder(), OrdinalEncoder()

    # Target variable
    df[target_variable] = df[target_variable].apply(lambda x: 0 if x == "Existing Customer" else 1)
    df.rename(columns={target_variable: "Attrition"}, inplace=True)

    # Ordinal variables
    scale_mapper = {"Unknown":1, "Uneducated":2, "High School":3, "College":4, "Graduate":5,
                    "Post-Graduate":6, "Doctorate":7}
    df["Education_Level"] = df["Education_Level"].replace(scale_mapper)

    scale_mapper = {"Unknown": 1, "Less than $40K": 2, "$40K - $60K": 3, "$60K - $80K": 4,
                    "$80K - $120K": 5, "$120K +":6}
    df["Income_Category"] = df["Income_Category"].replace(scale_mapper)

    # Label encoder
    df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
    df['Gender'] = le.fit_transform(df['Gender'])

    # Get dummies
    mask = df.dtypes == object
    categorical_variables = df.columns[mask]
    # df[categorical_variables] = ore.fit_transform(df[categorical_variables])
    df = pd.get_dummies(df, columns=categorical_variables, drop_first=True)

    return df


def _get_anova_fvalue(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    # Entre mayor sea el f1, quiere decir que la media entre las clases 0 y 1 de attrition, tiene una mayor variabilidad,
    # lo que quiere decir que esa variable si importa en en an?lisis
    f_scores = f_classif(x, y)[0]  # el [1] son los p-values.
    df_fscores = pd.DataFrame({'features': x.columns, 'score': f_scores})
    df_fscores = df_fscores.sort_values('score', ascending=False)

    return df_fscores


def _get_correlations(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    xcorr = data.corr().abs()
    xcorr = xcorr[xcorr > threshold].fillna(0)
    column1 = []
    column2 = []
    for idx in list(xcorr.index):
        for col in list(xcorr.columns):
            # la matriz es diagonal
            if idx == col:
                break
            if (xcorr.loc[idx, col] != 0):
                column1 = column1 + [idx]
                column2 = column2 + [col]
    df_fcorr = pd.DataFrame({'column1': column1, 'column2': column2})
    return df_fcorr


def _remove_columns_by_correlation(x: pd.DataFrame, df_most_correlated_cols: pd.DataFrame,
                                   df_anova_fscores: pd.DataFrame) -> pd.DataFrame:
    for idx in df_most_correlated_cols.index:
        column1 = df_most_correlated_cols.loc[idx, 'column1']
        column2 = df_most_correlated_cols.loc[idx, 'column2']
        score_column1 = df_anova_fscores.loc[df_anova_fscores['features'] == column1, 'score'].ravel()
        score_column2 = df_anova_fscores.loc[df_anova_fscores['features'] == column2, 'score'].ravel()
        if score_column1 > score_column2:
            df_most_correlated_cols.loc[idx, 'drop'] = column2
        else:
            df_most_correlated_cols.loc[idx, 'drop'] = column1
    drop_features = list(df_most_correlated_cols['drop'].unique())
    print("removed by correlation: ", drop_features)
    df_removed_columns = x.drop(columns=drop_features, axis=1)
    return df_removed_columns


def _remove_columns_by_fvalue(df_clean1: pd.DataFrame, df_anova_fscores: pd.DataFrame,
                              threshold: float) -> pd.DataFrame:
    df_anova_fscores = df_anova_fscores[df_anova_fscores['score'] > threshold]
    df_removed_columns = df_clean1[df_anova_fscores['features']]
    return df_removed_columns


def feature_selection_correlation_anova(df_encoded_data: pd.DataFrame, target: str, threshold: Dict) -> pd.DataFrame:
    x = df_encoded_data.drop(columns=[target])
    y = df_encoded_data[target]

    df_anova_fscores = _get_anova_fvalue(x, y)
    df_most_correlated_cols = _get_correlations(x, threshold['corr_threshold'])
    df_clean1 = _remove_columns_by_correlation(x, df_most_correlated_cols, df_anova_fscores)
    df_model_input = _remove_columns_by_fvalue(df_clean1, df_anova_fscores, threshold['fvalue_threshold'])
    df_model_input[target] = y

    return df_model_input