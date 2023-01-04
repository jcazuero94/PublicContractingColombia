from kedro.pipeline import Pipeline, node
from secop.pipelines.data_engineering.nodes import (
    secop_log,
    secop_extraction,
    clean_secop_int,
    clean_secop_2,
    clean_secop_2_cont,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                name="secop_2_log",
                func=secop_log,
                inputs=["params:code_secop2", "params:col_part_secop_2"],
                outputs="secop_2_log_in",
                tags=["data_engineering"],
            ),
            node(
                name="secop_2_cont_log",
                func=secop_log,
                inputs=["params:code_secop2_cont", "params:col_part_secop_2"],
                outputs="secop_2_cont_log_in",
                tags=["data_engineering"],
            ),
            node(
                name="secop_int_log",
                func=secop_log,
                inputs=["params:code_integrated", "params:col_part_secop_int"],
                outputs="secop_int_log_in",
                tags=["data_engineering"],
            ),
            # node(
            #     name="secop_2_extraction",
            #     func=secop_extraction,
            #     inputs=[
            #         "secop_2_log_in",
            #         "params:num_nits_to_extract",
            #         "params:code_secop2",
            #         "params:col_part_secop_2",
            #         "params:schema_secop_2_key",
            #     ],
            #     outputs=[
            #         "secop_2@spark",
            #         "secop_2_log_out",
            #     ],
            #     tags=["data_engineering"],
            # ),
            node(
                name="secop_2_cont_extraction",
                func=secop_extraction,
                inputs=[
                    "secop_2_cont_log_in",
                    "params:num_nits_to_extract",
                    "params:code_secop2_cont",
                    "params:col_part_secop_2",
                    "params:schema_secop_2_cont_key",
                ],
                outputs=[
                    "secop_2_cont@spark",
                    "secop_2_cont_log_out",
                ],
                tags=["data_engineering"],
            ),
            # node(
            #     name="secop_int_extraction",
            #     func=secop_extraction,
            #     inputs=[
            #         "secop_int_log_in",
            #         "params:num_nits_to_extract",
            #         "params:code_integrated",
            #         "params:col_part_secop_int",
            #         "params:schema_secop_int_key",
            #     ],
            #     outputs=[
            #         "secop_int",
            #         "secop_int_log_out",
            #     ],
            #     tags=["data_engineering"],
            # ),
            # node(
            #     name="clean_secop_int",
            #     func=clean_secop_int,
            #     inputs=["secop_int"],
            #     outputs="secop_int_clean",
            #     tags=["data_engineering"],
            # ),
            # node(
            #     name="clean_secop_2",
            #     func=clean_secop_2,
            #     inputs=["secop_2@pandas"],
            #     outputs="secop_2_clean",
            #     tags=["data_engineering"],
            # ),
            node(
                name="clean_secop_2_cont",
                func=clean_secop_2_cont,
                inputs=["secop_2_cont@pandas", "economia_departamentos"],
                outputs="secop_2_cont_clean",
                tags=["data_engineering"],
            ),
        ]
    )
