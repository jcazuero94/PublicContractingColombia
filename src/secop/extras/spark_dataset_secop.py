from kedro.extras.datasets.spark import SparkDataSet
from pyspark.sql import DataFrame


class SparkDataSetSECOP(SparkDataSet):
    def _save(self, data: DataFrame) -> None:
        """Modifies SparkDataSet for the save method be able to accept empty DataFrame skipping save"""
        if len(data.columns) > 1:
            super()._save(data)
