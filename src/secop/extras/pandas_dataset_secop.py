from kedro.extras.datasets.pandas import ParquetDataSet
import pandas as pd


class ParquetDataSetSecop(ParquetDataSet):
    def _load(self) -> pd.DataFrame:
        """Modifies parquet_dataset load method"""
        return self._load_from_pandas()
