"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    pipeline_lr = pipeline(
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
                outputs="model_lg",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model_lg","x_test","y_test"],
                outputs="classification_report_lr",
                name="evaluate_model_node",
            ),
        ],
        inputs="encoded_data",
        outputs="classification_report_lr",
        namespace="active_modelling_pipeline"
    )

    pipeline_svm = pipeline(
        [
            node(
                func=split_data,
                inputs=["encoded_data", "params:target", "params:model_options_svm"],
                outputs=["x_train", "y_train", "x_test", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["x_train", "y_train", "params:model_options_svm"],
                outputs="model_svm",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model_svm", "x_test", "y_test"],
                outputs="classification_report_svm",
                name="evaluate_model_node",
            ),
        ],
        inputs="encoded_data",
        outputs="classification_report_svm",
        namespace="candidate_modelling_pipeline_svm"
    )

    pipeline_rf = pipeline(
        [
            node(
                func=split_data,
                inputs=["encoded_data", "params:target", "params:model_options_rf"],
                outputs=["x_train", "y_train", "x_test", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["x_train", "y_train", "params:model_options_rf"],
                outputs="model_rf",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model_rf", "x_test", "y_test"],
                outputs="classification_report_rf",
                name="evaluate_model_node",
            ),
        ],
        inputs="encoded_data",
        outputs="classification_report_rf",
        namespace="candidate_modelling_pipeline_rf"
    )

    return pipeline(
        pipe=pipeline_lr + pipeline_svm + pipeline_rf,
        inputs="encoded_data",
        outputs=["classification_report_lr","classification_report_svm","classification_report_rf"],
        namespace="data_science",
    )
