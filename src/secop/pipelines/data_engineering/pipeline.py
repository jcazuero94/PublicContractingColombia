from kedro.pipeline import Pipeline, node
from secop.pipelines.data_engineering.nodes import secop_2_log


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=secop_2_log,
                inputs=None,
                outputs="secop2_log_in",
                name="secop_2_log",
                tags=["data_engineering"],
            ),
        ]
    )
