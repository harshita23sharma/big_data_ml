import click
from pipe import Pipe

from book_impact.transformers.drop_columns import DropColumns
from book_impact.transformers.get_concatenates_columns import \
    GetConcatenatedColumns
from book_impact.transformers.get_month_from_date import GetMonthFromDate
from book_impact.transformers.get_year_from_date import GetYearFromDate
from book_impact.transformers.save_to_parquet import SaveToParquet
from pyspark.sql import DataFrame, SparkSession

from pyspark.sql import DataFrame, SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import PipelineModel
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from timeit import default_timer as timer
from datetime import timedelta


class ParquetDataFrame(DataFrame):
    """DataFrame related to partquet files."""

    def __init__(self, path: str, spark: SparkSession):
        super(ParquetDataFrame, self).__init__(spark.read.parquet(path)._jdf, spark)



@click.command()
@click.option("-phase", "--phase", default="train", type=str, help="train/test")
@click.option("-m", "--master-url", default="local[2]", type=str, help="master url")
@click.option(
    "-i", "--input-path", default="data/processed/train/features", type=str
)
# @click.argument("input_path")
def preprocess_data(phase, master_url, input_path):
    spark = (
        SparkSession.builder.master(master_url).appName("book_impact_preprocessing").getOrCreate()
    )
    features_df = ParquetDataFrame(input_path, spark)

    test_data_frac = 0.1
    test_features_df, train_features_df = features_df.randomSplit([test_data_frac, 1 - test_data_frac])
    label_col = 'Impact'
    assembler_estimator = Pipeline(stages=[
        StringIndexer(inputCol=column_name, outputCol= output_col_name)
        StringIndexer(inputCol='title_desc', handleInvalid='keep', outputCol='title_desc'),
        VectorAssembler(inputCols=['pickup_cell_6_idx', 'dropoff_cell_6_idx', 'distance', 'month', 'day_of_month',
                                   'day_of_week', 'hour', 'requests_pickup_cell', 'requests_dropoff_cell'],
                        outputCol="features")
    ])

    dt_estimator = DecisionTreeRegressor(maxDepth=5, featuresCol='features', labelCol=label_col, maxBins=32)
    rf_estimator = RandomForestRegressor(maxDepth=5, numTrees=40, featuresCol='features', labelCol=label_col)

    pipeline = Pipeline(stages=[])
    dt_stages = [assembler_estimator, dt_estimator]
    rf_stages = [assembler_estimator, rf_estimator]

    dt_grid = ParamGridBuilder().baseOn({pipeline.stages: dt_stages}) \
        .addGrid(dt_estimator.maxDepth, [2, 5, 7, 9]) \
        .build()

    rf_grid = ParamGridBuilder().baseOn({pipeline.stages: rf_stages}) \
        .addGrid(rf_estimator.maxDepth, [5, 7]) \
        .addGrid(rf_estimator.numTrees, [10, 20]) \
        .build()

    grid = dt_grid + rf_grid

    eval_metric = 'mae'
    folds = 3
    print(f'Preparing {eval_metric} evaluator and {folds}-fold cross-validator...')
    mae_evaluator = RegressionEvaluator(metricName=eval_metric, labelCol=label_col)
    cross_val = CrossValidator(estimatorParamMaps=grid, estimator=pipeline,
                               evaluator=mae_evaluator, numFolds=folds, parallelism=4)

    print(f'Searching for parameters...')
    start = timer()
    cross_val_model = cross_val.fit(train_features_df)
    end = timer()
    print(f'Search complete, duration: {timedelta(seconds=end - start)}')
    print(f'Best model: {cross_val_model.bestModel.stages[1]}')

    predictions_df = cross_val_model.transform(test_features_df)
    mae_cv = RegressionEvaluator(labelCol=label_col, metricName=eval_metric).evaluate(predictions_df)
    print(f'Best model MAE: {mae_cv}')

    print(f'Best model parameters:')
    for item in cross_val_model.bestModel.stages[1].extractParamMap().items():
        print(f'- {item[0]}: {item[1]}')

    spark.stop()


if __name__ == "__main__":
    preprocess_data()
