from kedro.pipeline import Pipeline, node
from secop.pipelines.data_science.nodes import (
    split_contract_value,
    prepare_clusters_contract,
    train_rnn,
    prepare_dfs_providers,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                name="split_contract_value",
                func=split_contract_value,
                inputs=["secop_2_cont_clean", "params:features_text"],
                outputs=[
                    "train_contract_value",
                    "cv_contract_value",
                    "test_contract_value",
                    "train_contract_value_rnn",
                    "cv_contract_value_rnn",
                    "test_contract_value_rnn",
                ],
                tags=["data_science"],
            ),
            # node(
            #     name="prepare_clusters_contract",
            #     func=prepare_clusters_contract,
            #     inputs=["secop_2_cont_clean", "params:features_text"],
            #     outputs="df_clusters_contract",
            #     tags=["data_science"],
            # ),
            # node(
            #     name="train_rnn",
            #     func=train_rnn,
            #     inputs=[
            #         "train_contract_value_rnn",
            #         "cv_contract_value_rnn",
            #         "params:rnn_from_checkpoint",
            #     ],
            #     outputs=None,
            #     tags=["data_science"],
            # ),
            node(
                name="prepare_dfs_providers",
                func=prepare_dfs_providers,
                inputs="secop_2_cont_clean",
                outputs=["df_personas", "df_empresas"],
                tags=["data_science"],
            ),
        ]
    )
