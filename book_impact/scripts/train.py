from datetime import timedelta
from timeit import default_timer as timer

import click
import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

MLFLOW_EXPERIMENT_NAME = "Model Training"

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


@click.command()
@click.option("-phase", "--phase", default="train", type=str, help="train/test")
@click.option("-m", "--master-url", default="local[2]", type=str, help="master url")
@click.option(
    "-i", "--input-path", default="data/processed/train/features", type=str
)
# @click.argument("input_path")
def train_model(phase, master_url, input_path):
    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id
    except:
        print("Create a new mlflow experiment")
        experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(experiment_id=experiment_id):
        spark = (
            SparkSession.builder.master(master_url).appName("book_impact_train").getOrCreate()
        )
        features_df = ParquetDataFrame(input_path, spark)
        mlflow.log_param("master_url", master_url)
        mlflow.log_param("input_path", input_path)
        

        test_data_frac = 0.1
        features_df2 = features_df.na.drop(subset=CAT_COLUMNS).sample(0.1)
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

        rf_estimator = RandomForestRegressor(maxDepth=7, numTrees=20, minInstancesPerNode=1,featuresCol='features', labelCol=label_col)

        pipeline = Pipeline(stages=[assembler_estimator, rf_estimator])

        eval_metric = 'mae'
        folds = 3
        print(f'Preparing {eval_metric} evaluator and {folds}-fold cross-validator...')
        mae_evaluator = RegressionEvaluator(metricName=eval_metric, labelCol=label_col)

        print(f'Searching for parameters...')
        start = timer()
        model = pipeline.fit(train_features_df)
        
        
        model_path = 'model/book_impact_model'

        print(f'Saving model to {model_path}')
        model.write().overwrite().save(model_path)
        print(f'Model saved...')

        predictions_df = model.transform(test_features_df)
        mae_cv = RegressionEvaluator(labelCol=label_col, metricName=eval_metric).evaluate(predictions_df)
        print(f'Best model MAE: {mae_cv}')

        mlflow.log_metric("Model MAE on test set", mae_cv)
        end = timer()
        time_taken = timedelta(seconds=end - start)
        mlflow.log_param("time_taken", time_taken)
        print(f'Search complete, duration: {time_taken}')
        spark.stop()
        mlflow.end_run()


if __name__ == "__main__":
    train_model()
