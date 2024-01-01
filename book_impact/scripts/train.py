from datetime import timedelta
from timeit import default_timer as timer

import click
from pipe import Pipe
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import (DecisionTreeRegressor, GBTRegressor,
                                   RandomForestRegressor)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import StringType

from book_impact.transformers.drop_columns import DropColumns
from book_impact.transformers.get_concatenates_columns import \
    GetConcatenatedColumns
from book_impact.transformers.get_month_from_date import GetMonthFromDate
from book_impact.transformers.get_year_from_date import GetYearFromDate
from book_impact.transformers.save_to_parquet import SaveToParquet


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
# NUMERIC_COLUMNS = ['publisher_', 'categories_onehot', 'publishedDate_year_onehot', 'publishedDate_month_onehot']]



@click.command()
@click.option("-phase", "--phase", default="train", type=str, help="train/test")
@click.option("-m", "--master-url", default="local[1]", type=str, help="master url")
@click.option(
    "-i", "--input-path", default="data/processed/train/features", type=str
)
# @click.argument("input_path")
def preprocess_data(phase, master_url, input_path):
    spark = (
        SparkSession.builder.master(master_url).appName("book_impact_train").getOrCreate()
    )
    features_df = ParquetDataFrame(input_path, spark)

    test_data_frac = 0.1
    features_df2 = features_df.na.drop(subset=CAT_COLUMNS).sample(0.05)
    print("features_df2:", features_df2.count())
    transform_empty = udf(lambda s: "NA" if s == "" else s, StringType())
    for col in CAT_COLUMNS:
        features_df2 = features_df2.withColumn(col, transform_empty(col))

    test_features_df, train_features_df = features_df2.randomSplit([test_data_frac, 1 - test_data_frac])
    label_col = 'Impact'
    stages = []
    for col_name in CAT_COLUMNS:
        stages.append(pipeline_fit_category_to_one_hot_encode(col_name, col_name+"_numeric"))
        stages.append(pipeline_transform_category_to_one_hot_encode(col_name+"_numeric", col_name +'_onehot'))
    stages.append(VectorAssembler(inputCols=ONE_HOT_ENCODED_COLS,
                        outputCol="features"))
    assembler_estimator = Pipeline(stages = stages)

    dt_estimator = DecisionTreeRegressor(maxDepth=5, featuresCol='features', labelCol=label_col, maxBins=32)
    rf_estimator = RandomForestRegressor(maxDepth=5, numTrees=4, featuresCol='features', labelCol=label_col)

    pipeline = Pipeline(stages=[])
    dt_stages = [assembler_estimator, dt_estimator]
    rf_stages = [assembler_estimator, rf_estimator]

    dt_grid = ParamGridBuilder().baseOn({pipeline.stages: dt_stages}) \
        .addGrid(dt_estimator.maxDepth, [2]) \
        .build()

    rf_grid = ParamGridBuilder().baseOn({pipeline.stages: rf_stages}) \
        .addGrid(rf_estimator.maxDepth, [5, 7]) \
        .addGrid(rf_estimator.numTrees, [10]) \
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
