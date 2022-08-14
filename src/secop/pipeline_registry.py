"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from secop.pipelines import data_engineering as de
from secop.pipelines import data_science as ds


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_engineering_pipeline = de.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    return {
        "__default__": data_engineering_pipeline + data_science_pipeline,
        "data_engineering": data_engineering_pipeline,
        "data_science": data_science_pipeline,
    }
