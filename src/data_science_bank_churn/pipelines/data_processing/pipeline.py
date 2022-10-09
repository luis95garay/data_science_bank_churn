"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import remove_unnecessary_columns, reduce_categorical_column_options, handle_outliers, treat_skewed_columns, encode_categorical_columns

def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline(
        [
            node(
                func=remove_unnecessary_columns,
                inputs=["BankChurners", "params:delete_columns"],
                outputs="BankChurners_removed_columns",
                name="remove_incomplete_columns",
            ),
            node(
                func=reduce_categorical_column_options,
                inputs=["BankChurners_removed_columns",'params:reduce_options_columns'],
                outputs="BankChurners_reduced_columns",
                name="reduce_columns_options",
            ),
            node(
                func=handle_outliers,
                inputs=["BankChurners_reduced_columns", 'params:outliers_columns'],
                outputs="BankChurners_without_outliers",
                name="handle_outliers",
            ),
            node(
                func=treat_skewed_columns,
                inputs=["BankChurners_without_outliers", 'params:skewed_columns'],
                outputs="BankChurners_without_skew",
                name="treat_skew",
            ),
            node(
                func=encode_categorical_columns,
                inputs=["BankChurners_without_skew",'params:target_variable'],
                outputs="encoded_data",
                name="encode_categorical_columns",
            ),
        ]
    )
    return pipeline(
        pipe=pipeline_instance,
        inputs="BankChurners",
        outputs="encoded_data",
        namespace="data_processing",
    )
