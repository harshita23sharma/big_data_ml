import click
# import sparknlp
from pipe import Pipe
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import IDF, HashingTF, Tokenizer

from book_impact.transformers.drop_columns import DropColumns
from book_impact.transformers.encode_text_data import EncodeTextData
from book_impact.transformers.get_concatenates_columns import \
    GetConcatenatedColumns
from book_impact.transformers.get_month_from_date import GetMonthFromDate
from book_impact.transformers.get_year_from_date import GetYearFromDate
from book_impact.transformers.save_to_parquet import SaveToParquet

# from sparknlp.annotator import *
# from sparknlp.base import *


# from sparknlp. import *


COLS_TO_DROP = ["_c0", "Title", "description", "publishedDate", "publisher_count"]
from pyspark.sql import DataFrame, SparkSession


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
    "-i", "--input-path", default="/Users/harshita/Downloads/books_task.csv", type=str
)
@click.option("-t", "--text-pipeline-path", default="data/processed/train/text_features", type=str)
# @click.argument("input_path")
def preprocess_data(phase, master_url, input_path, text_pipeline_path):
    spark = (
        SparkSession.builder.master(master_url).appName("book_impact_eda").getOrCreate()
    )
    # spark= sparknlp.start()

    # document_assembler = DocumentAssembler()\
    # .setInputCol("title_desc")\
    # .setOutputCol("document")

    # tokenizer = Tokenizer().setInputCols(["document"])\
    # .setOutputCol("token")
    
    # word_embeddings = BertEmbeddings.pretrained('small_bert_L2_768', 'en')\
    # .setInputCols(["document", "token"])\
    # .setOutputCol("embeddings")


    # bert_pipeline = Pipe(
    # [
    #     document_assembler,
    #     tokenizer,
    #     word_embeddings
    # ]
    # )
    tokenizer = Tokenizer(inputCol="title_desc", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    pipeline = Pipeline(stages=[tokenizer, hashingTF])

    
    df = Pipe(
        [
            GetYearFromDate("publishedDate"),
            GetMonthFromDate("publishedDate"),
            GetConcatenatedColumns(["title", "description"]),
            # EncodeTextData(text_pipeline_path,"title_desc")
            # bert_pipeline("title_description").stages,
            # DropColumns(inputCols=COLS_TO_DROP),
            # SaveToParquet(f"data/processed/{phase}/features"),
            # document_assembler,
            # tokenizer,
            # word_embeddings
        ]
    ).transform(CsvDataFrame(input_path, spark))

    # # Fit the pipeline to training documents (TEXT COLUMN).
    training = pipeline.fit(df.select("title_desc").fillna("Blank"))
    # Save Text processing pipeline for inferencing on test / new data
    training.write().overwrite().save(text_pipeline_path)

    #Load Text processing pipeline for inferencing on test / new data
    inference_text_model = Pipe(PipelineModel.load(text_pipeline_path).stages)
    dataset = inference_text_model.transform(df.select("title_desc").fillna("Blank"))
    print(dataset.show(2))
    print(f"Saved {dataset.count()} rows of {phase} inputs")
    spark.stop()



if __name__ == "__main__":
    preprocess_data()
