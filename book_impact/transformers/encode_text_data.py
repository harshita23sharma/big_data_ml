from pyspark.ml import PipelineModel, Transformer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class EncodeTextData(Transformer):
    """Encode or embed text data."""

    def __init__(self, text_pipeline_path:str, column:str):
        super(EncodeTextData, self).__init__()
        self._col = column
        self.inference_text_model = PipelineModel.load(text_pipeline_path)


    def _transform(self, dataset: DataFrame) -> DataFrame:
        dataset = dataset.withColumn('id', F.monotonically_increasing_id())
        dataset2 = self.inference_text_model.transform(dataset.select("title_desc").fillna("Blank"))
        # Add same id column to dataset2
        dataset2 = dataset2.withColumn('id', F.monotonically_increasing_id())
        # Now join on the id column
        merged_df = dataset.join(dataset2.select(["id","text_features"]), on='id', how='left')        
        return merged_df.drop("id")
