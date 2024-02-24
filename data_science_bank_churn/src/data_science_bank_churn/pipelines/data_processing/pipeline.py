"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    handle_outliers, transform_target, get_preprocessor
)


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline(
        [
            node(
                func=transform_target,
                # inputs=["BankChurners", "params:delete_columns"],
                inputs=["BankChurners"],
                outputs="rename_bankchurners",
                name="rename_target",
            ),
            node(
                func=handle_outliers,
                inputs=[
                    "rename_bankchurners",
                    'params:outliers_columns'
                ],
                outputs="model_input",
                name="handle_outliers",
            ),
            node(
                func=get_preprocessor,
                inputs=["model_input"],
                outputs="preprocessor",
                name="feature_selection",
            ),
        ]
    )
    return pipeline(
        pipe=pipeline_instance,
        inputs="BankChurners",
        outputs=["model_input", "preprocessor"],
        namespace="data_processing",
    )
