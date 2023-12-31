from typing import List

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols
from pyspark.sql import DataFrame


class DropColumns(
    Transformer,
    HasInputCols,
):
    """Transformer that drops specified columns."""

    @keyword_only
    def __init__(self, inputCols: List[str]):
        super(DropColumns, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols: List[str]):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset: DataFrame):
        return dataset.drop(*self.getInputCols())
