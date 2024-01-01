import click
from pipe import Pipe
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (IDF, HashingTF, StringIndexer, Tokenizer,
                                Word2Vec)
from pyspark.sql import DataFrame, SparkSession

from book_impact.transformers.get_concatenates_columns import \
    GetConcatenatedColumns


class CsvDataFrame(DataFrame):
    """DataFrame related to a csv file."""

    def __init__(self, path: str, spark: SparkSession, header: bool = True):
        super(CsvDataFrame, self).__init__(
            spark.read.option("header", str(header).lower())\
                .option("inferSchema", True).option("escape", '"').csv(path)._jdf, spark
        )


@click.command()
@click.option("-phase", "--phase", default="train", type=str, help="train/test")
@click.option("-m", "--master-url", default="local[2]", type=str, help="master url")
@click.option(
    "-i", "--input-path", default="data/raw/books_task.csv", type=str
)
@click.option("-t", "--text-pipeline-path", default="data/processed/train/text_features", type=str)
def train_tokeniser(phase, master_url, input_path, text_pipeline_path):
    spark = (
        SparkSession.builder.master(master_url).appName("book_impact_train_tokenizer").getOrCreate()
    )

    tokenizer = Tokenizer(inputCol="title_desc", outputCol="words")
    # word2Vec = Word2Vec(vectorSize=30, minCount=0, inputCol="words", outputCol="text_features")

    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="tf", numFeatures=200)
    idf = IDF(inputCol='tf', outputCol="text_features", minDocFreq=5) #minDocFreq: remove sparse terms
    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf])
    # pipeline = Pipeline(stages=[tokenizer, word2Vec])
    
    #Just concate title and description columns
    df = Pipe(
        [
            GetConcatenatedColumns(["title", "description"]),
        ]
    ).transform(CsvDataFrame(input_path, spark))

    # Fit the pipeline to training documents (TEXT COLUMN).
    training = pipeline.fit(df.select("title_desc").fillna("Blank"))
    # Save Text processing pipeline for inferencing on test / new data
    training.write().overwrite().save(text_pipeline_path)

    # #Load Text processing pipeline for inferencing on test / new data
    inference_text_model = Pipe(PipelineModel.load(text_pipeline_path).stages)
    dataset = inference_text_model.transform(df.select("title_desc").fillna("Blank"))
    print(f"Saved {dataset.count()} rows of {phase} inputs")
    spark.stop()



if __name__ == "__main__":
    train_tokeniser()
