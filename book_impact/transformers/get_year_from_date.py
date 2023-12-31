import pyspark.sql.functions as fn
from pyspark.ml import Transformer
from pyspark.sql import DataFrame


class GetYearFromDate(Transformer):
    """Extract year from date."""

    def __init__(self, column: str):
        super(GetYearFromDate, self).__init__()
        self._col = column

    def _transform(self, dataset: DataFrame) -> DataFrame:
        dataset = dataset.withColumn(
            f"{self._col}_year", fn.regexp_extract(self._col, r"\b(\d{4})\b", 1)
        )
        return dataset
