"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats.mstats import normaltest
from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


def transform_target(df: pd.DataFrame):
    target_variable = 'Attrition_Flag'
    df[target_variable] = df[target_variable].apply(
        lambda x: 0 if x == "Existing Customer" else 1
    )
    df.rename(columns={target_variable: "Attrition"}, inplace=True)
    return df


def handle_outliers(
    df: pd.DataFrame,
    outliers_columns: List
) -> pd.DataFrame:
    """
    Handles outliers in specified columns of a DataFrame based on statistical
    tests and boundary criteria.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - outliers_columns (List): A list containing information about
      columns with outliers.

    Returns:
    - pd.DataFrame: The DataFrame with outliers removed.
    """
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
        outliers = df[
            (df[col] < lower_boundary) | (df[col] > uppper_boundary)
        ].index.tolist()

        if len(outliers) > 0:
            df.drop(outliers, axis=0, inplace=True)
            transformed_columns.append((col, len(outliers)))

    df.reset_index(inplace=True, drop=True)
    if len(transformed_columns) > 0:
        print("Outliers deleted: ", transformed_columns)
        print("Shape after removing: ", df.shape)
    else:
        print("There are not outliers")
    return df


def get_preprocessor(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Encodes categorical columns in a DataFrame using various encoding
    techniques.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_variable (str): The name of the target variable for binary
      encoding.

    Returns:
    - pd.DataFrame: The DataFrame with encoded categorical columns.
    """
    target_variable = 'Attrition'
    x = df.drop(columns=[target_variable])


    education_order = ['Unknown', 'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate']
    income_order = ["Unknown", "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"]


    # Create a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('Custom_education', OrdinalEncoder(categories=[education_order]), ['Education_Level']),
            ('Custom_income', OrdinalEncoder(categories=[income_order]), ['Income_Category']),
            ('MinMax', MinMaxScaler(), ['Customer_Age', 'Months_on_book', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Trans_Amt']),
            ('Ordinal', OrdinalEncoder(), ['Marital_Status', 'Gender']),
            ('onehot', OneHotEncoder(), ['Card_Category'])
        ],
        remainder='passthrough', # Leave the other columns unchanged

    )

    # Label encoder
    preprocessor.fit(x) 

    return preprocessor


def remove_incomplete_rows(
    df: pd.DataFrame
) -> pd.DataFrame:
    missing_data_cols = [
        col for col in df.columns.tolist() if len(
            df[pd.isnull(df[col])]
        ) > 0
    ]
    for column in missing_data_cols:
        null_values = df[pd.isnull(df[column])].index.tolist()
        df.drop(null_values, axis=0, inplace=True)
        # print(column, '\t', len(null_values), '\t', df[column].dtypes)
    print("Incomplete deleted rows: ", missing_data_cols)
    return df


def reduce_categorical_column_options(
        df: pd.DataFrame,
        reduce_columns: Dict
) -> pd.DataFrame:
    """
    Reduces the number of options in categorical columns of a DataFrame based
    on specified criteria.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - reduce_columns (Dict): A dictionary containing information about columns
      to be reduced.
        - 'exclude' (List[str]): List of column names to be excluded from
          reduction.
        - 'threshold' (float): Threshold for reducing categories. Categories
          with a frequency below this threshold will be replaced with
          'others'.

    Returns:
    - pd.DataFrame: The DataFrame with reduced categorical options.
    """
    df_data_red = df.copy()
    mask = df.dtypes == object
    categorical_cols = df.columns[mask]
    combined_col_names = []
    for col in categorical_cols:
        if col not in reduce_columns['exclude']:
            val_counts = df[col].value_counts()
            replace_cats = list(
                val_counts[
                    (val_counts / val_counts.sum()) < reduce_columns['threshold']
                ].index
            )
            if len(replace_cats) > 0:
                df_data_red[col] = df_data_red.replace(replace_cats, 'others')[col]
                combined_col_names.append(col)

    if len(combined_col_names) > 0:
        print("Reduced columns: ", combined_col_names)
    else:
        print("Not reduced columns")

    return df_data_red


def handle_outliers(
    df: pd.DataFrame,
    outliers_columns: Dict
) -> pd.DataFrame:
    """
    Handles outliers in specified columns of a DataFrame based on statistical
    tests and boundary criteria.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - outliers_columns (Dict): A dictionary containing information about
      columns with outliers.
        - Keys: Column names with outliers.
        - Values: Not used. Can be an empty dictionary or any placeholder.

    Returns:
    - pd.DataFrame: The DataFrame with outliers removed.
    """
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
        outliers = df[
            (df[col] < lower_boundary) | (df[col] > uppper_boundary)
        ].index.tolist()

        if len(outliers) > 0:
            df.drop(outliers, axis=0, inplace=True)
            transformed_columns.append((col, len(outliers)))

    if len(transformed_columns) > 0:
        print("Outliers deleted: ", transformed_columns)
        print("Shape after removing: ", df.shape)
    else:
        print("There are not outliers")
    return df


def treat_skewed_columns(
    df: pd.DataFrame,
    skewed_columns: Dict
) -> pd.DataFrame:
    """
    Treats skewed numerical columns in a DataFrame using specified
    transformation methods.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - skewed_columns (Dict): A dictionary containing information about columns
      and transformation methods.
        - 'exclude_columns' (List[str]): List of column names to be excluded
          from skewness treatment.
        - 'method' (str): Transformation method. Options: "boxcox" or "log".

    Returns:
    - pd.DataFrame: The DataFrame with treated skewed columns.
    """
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
                # print(col, p_value)
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


def encode_categorical_columns(
    df: pd.DataFrame,
    target_variable: str
) -> pd.DataFrame:
    """
    Encodes categorical columns in a DataFrame using various encoding
    techniques.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_variable (str): The name of the target variable for binary
      encoding.

    Returns:
    - pd.DataFrame: The DataFrame with encoded categorical columns.
    """
    # lb, le, ore = LabelBinarizer(), LabelEncoder(), OrdinalEncoder()
    le = LabelEncoder()

    # Target variable
    df[target_variable] = df[target_variable].apply(
        lambda x: 0 if x == "Existing Customer" else 1
    )
    df.rename(columns={target_variable: "Attrition"}, inplace=True)

    # Ordinal variables
    scale_mapper = {
        "Unknown": 1, "Uneducated": 2, "High School": 3,
        "College": 4, "Graduate": 5, "Post-Graduate": 6, "Doctorate": 7
    }
    df["Education_Level"] = df["Education_Level"].replace(scale_mapper)

    scale_mapper = {
        "Unknown": 1, "Less than $40K": 2, "$40K - $60K": 3, "$60K - $80K": 4,
        "$80K - $120K": 5, "$120K +": 6
    }
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


def _get_anova_fvalue(
    x: pd.DataFrame,
    y: pd.Series
) -> pd.DataFrame:
    # Entre mayor sea el f1, quiere decir que la media entre las
    # clases 0 y 1 de attrition, tiene una mayor variabilidad,
    # lo que quiere decir que esa variable si importa en en an?lisis
    f_scores = f_classif(x, y)[0]  # el [1] son los p-values.
    df_fscores = pd.DataFrame({'features': x.columns, 'score': f_scores})
    df_fscores = df_fscores.sort_values('score', ascending=False)

    return df_fscores


def _get_correlations(
    data: pd.DataFrame,
    threshold: float
) -> pd.DataFrame:
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


def _remove_columns_by_correlation(
    x: pd.DataFrame,
    df_most_correlated_cols: pd.DataFrame,
    df_anova_fscores: pd.DataFrame
) -> pd.DataFrame:
    for idx in df_most_correlated_cols.index:
        column1 = df_most_correlated_cols.loc[idx, 'column1']
        column2 = df_most_correlated_cols.loc[idx, 'column2']
        score_column1 = df_anova_fscores.loc[
            df_anova_fscores['features'] == column1, 'score'
        ].ravel()
        score_column2 = df_anova_fscores.loc[
            df_anova_fscores['features'] == column2, 'score'
        ].ravel()
        if score_column1 > score_column2:
            df_most_correlated_cols.loc[idx, 'drop'] = column2
        else:
            df_most_correlated_cols.loc[idx, 'drop'] = column1
    drop_features = list(df_most_correlated_cols['drop'].unique())
    print("removed by correlation: ", drop_features)
    df_removed_columns = x.drop(columns=drop_features, axis=1)
    return df_removed_columns


def _remove_columns_by_fvalue(
    df_clean1: pd.DataFrame,
    df_anova_fscores: pd.DataFrame,
    threshold: float
) -> pd.DataFrame:
    df_anova_fscores = df_anova_fscores[df_anova_fscores['score'] > threshold]
    df_removed_columns = df_clean1[df_anova_fscores['features']]
    return df_removed_columns


def feature_selection_correlation_anova(
    df_encoded_data: pd.DataFrame,
    target: str, threshold: Dict
) -> pd.DataFrame:
    """
    Performs feature selection based on correlation and ANOVA F-value
    criteria.

    Parameters:
    - df_encoded_data (pd.DataFrame): The input DataFrame with encoded
      features.
    - target (str): The name of the target variable.
    - threshold (Dict): A dictionary containing threshold values for
      feature selection.
        - 'corr_threshold' (float): Threshold for correlation coefficient.
        - 'fvalue_threshold' (float): Threshold for ANOVA F-value.

    Returns:
    - pd.DataFrame: The DataFrame with selected features based on
      correlation and ANOVA F-value.
    """
    x = df_encoded_data.drop(columns=[target])
    y = df_encoded_data[target]

    df_anova_fscores = _get_anova_fvalue(x, y)
    df_most_correlated_cols = _get_correlations(
        x, threshold['corr_threshold']
    )
    df_clean1 = _remove_columns_by_correlation(
        x, df_most_correlated_cols, df_anova_fscores
    )
    df_model_input = _remove_columns_by_fvalue(
        df_clean1, df_anova_fscores, threshold['fvalue_threshold']
    )
    df_model_input[target] = y

    return df_model_input
