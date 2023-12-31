from typing import List

from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col


class GetConcatenatedColumns(Transformer):
    """Extract year from date."""

    def __init__(self, columns: List[str]):
        super(GetConcatenatedColumns, self).__init__()
        self._cols = columns

    def _transform(self, dataset: DataFrame) -> DataFrame:
        dataset.withColumn(self._cols[0], F.when(F.col(self._cols[0]).isNull(), '').otherwise(F.col(self._cols[0])))
        dataset.withColumn(self._cols[0], F.when(F.col(self._cols[1]).isNull(), '').otherwise(F.col(self._cols[1])))
        dataset = dataset.withColumn(
            f"title_desc", F.concat(col(self._cols[0]), col(self._cols[1]))
        )
        return dataset
