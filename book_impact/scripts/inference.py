from datetime import timedelta
from timeit import default_timer as timer

import click
import mlflow
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

from book_impact.transformers.drop_columns import DropColumns
from book_impact.transformers.save_to_parquet import SaveToParquet

MLFLOW_EXPERIMENT_NAME = "Model Inferencing"
COLS_TO_DROP = ["_c0", "Title", "description", "publishedDate", "publisher_count"]


class ParquetDataFrame(DataFrame):
    """DataFrame related to partquet files."""

    def __init__(self, path: str, spark: SparkSession):
        super(ParquetDataFrame, self).__init__(spark.read.parquet(path)._jdf, spark)


def pipeline_fit_category_to_one_hot_encode(column_name, output_col_name):
    indexer = StringIndexer(inputCol=column_name, outputCol= output_col_name, handleInvalid="keep")
    return indexer

def pipeline_transform_category_to_one_hot_encode(column_name, output_col_name):
    encoder = OneHotEncoder(inputCols=[column_name], outputCols=[output_col_name], handleInvalid="keep")
    return encoder

CAT_COLUMNS = ['publisher',
                    'categories',
                    'publishedDate_year',
                    'publishedDate_month'
                    ]
ONE_HOT_ENCODED_COLS = ['publisher_onehot', 'categories_onehot', 'publishedDate_year_onehot', 'publishedDate_month_onehot']
VECTOR_COL = ["text_features"]

@click.command()
@click.option("-phase", "--phase", default="train", type=str, help="train/test")
@click.option("-m", "--master-url", default="local[2]", type=str, help="master url")
@click.option(
    "-t", "--test-data-path", default="data/processed/test/features", type=str
)
@click.option("--model-path", default="model/book_impact_model", type=str)
@click.option("--prediction-path", default="data/predictions/", type=str)
def inference_model(phase, master_url, test_data_path, model_path, prediction_path):
    spark = (
        SparkSession.builder.master(master_url).appName("book_impact_inference").getOrCreate()
    )
    model = PipelineModel.load(model_path)
    predicted_df = PipelineModel([
            model,
            DropColumns(inputCols=['features']),
            SaveToParquet(prediction_path)
        ]).transform(ParquetDataFrame(test_data_path, spark))

    predicted_df.select("Impact","prediction").show(20)
    spark.stop()


if __name__ == "__main__":
    inference_model()
