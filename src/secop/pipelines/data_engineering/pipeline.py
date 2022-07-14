from kedro.pipeline import Pipeline, node
from secop.pipelines.data_engineering.nodes import (
    secop_2_log,
    secop_2_extraction,
    secop_int_log,
    secop_int_extraction,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=secop_2_log,
                inputs=None,
                outputs="secop_2_log_in",
                name="secop_2_log",
                tags=["data_engineering"],
            ),
            node(
                func=secop_int_log,
                inputs=None,
                outputs="secop_int_log_in",
                name="secop_int_log",
                tags=["data_engineering"],
            ),
            node(
                func=secop_2_extraction,
                inputs=["secop_2_log_in"],
                outputs=["secop_2_log_out", "secop_2"],
                name="secop_2_extraction",
                tags=["data_engineering"],
            ),
            node(
                func=secop_int_extraction,
                inputs=["secop_int_log_in"],
                outputs=["secop_int_log_out", "secop_int"],
                name="secop_int_extraction",
                tags=["data_engineering"],
            ),
        ]
    )
