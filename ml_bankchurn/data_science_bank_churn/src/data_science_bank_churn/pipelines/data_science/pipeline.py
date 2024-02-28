"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_dataset, train_model


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_data_science = pipeline(
        [
            node(
                func=split_dataset,
                inputs=["model_input", "preprocessor"],
                outputs=["x_train", "y_train", "x_test", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["x_train", "y_train", "x_test", "y_test"],
                outputs="best_model",
                name="train_model_node",
            )
        ],
    )

    return pipeline(
        pipe=pipeline_data_science,
        inputs=["model_input", "preprocessor"],
        outputs="best_model",
        namespace="data_science",
    )
