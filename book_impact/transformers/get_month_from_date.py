import pyspark.sql.functions as fn
from pyspark.ml import Transformer
from pyspark.sql import DataFrame


class GetMonthFromDate(Transformer):
    """Extract year from date."""

    def __init__(self, column: str):
        super(GetMonthFromDate, self).__init__()
        self._col = column

    def _transform(self, dataset: DataFrame) -> DataFrame:
        dataset.withColumn(
            f"{self._col}_month", fn.regexp_extract(self._col, r"(?<=-)(\d{2})", 0)
        )
        return dataset
