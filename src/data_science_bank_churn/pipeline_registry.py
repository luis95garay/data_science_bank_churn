"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from data_science_bank_churn.pipelines import data_processing
from data_science_bank_churn.pipelines import data_science


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_processing_pipeline = data_processing.create_pipeline()
    data_science_pipeline = data_science.create_pipeline()

    return {
        "__default__": data_processing_pipeline + data_science_pipeline,
        "data_processing": data_processing_pipeline,
        "data_science": data_science_pipeline,
    }
