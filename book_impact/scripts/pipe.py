from typing import List

from pyspark.ml import PipelineModel, Transformer
from pyspark.sql import DataFrame


class Pipe(Transformer):
    """Conditional pipeline which runs one or another list of transformers based on condition"""

    def __init__(self, stages: List[Transformer]):
        super(Pipe, self).__init__()
        self._pipeline = PipelineModel(stages)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return self._pipeline.transform(dataset)
