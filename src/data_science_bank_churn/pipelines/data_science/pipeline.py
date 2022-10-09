"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline(
        [
            node(
                func=split_data,
                inputs=["encoded_data","params:target","params:model_options_lg"],
                outputs=["x_train", "y_train", "x_test", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["x_train", "y_train", "params:model_options_lg"],
                outputs="model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model","x_test","y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )
    ds_pipeline_1 = pipeline(
        pipe=pipeline_instance,
        inputs="encoded_data",
        namespace="active_modelling_pipeline",
    )
    ds_pipeline_2 = pipeline(
        pipe=pipeline_instance,
        inputs="encoded_data",
        namespace="candidate_modelling_pipeline",
        parameters={"params:model_options_lg": "params:model_options_svm"},
    )
    return pipeline(
        pipe=ds_pipeline_1 + ds_pipeline_2,
        inputs="encoded_data",
        namespace="data_science",
    )
