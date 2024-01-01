import click
from pipe import Pipe

from book_impact.transformers.drop_columns import DropColumns
from book_impact.transformers.get_concatenates_columns import \
    GetConcatenatedColumns
from book_impact.transformers.get_month_from_date import GetMonthFromDate
from book_impact.transformers.get_year_from_date import GetYearFromDate
from book_impact.transformers.save_to_parquet import SaveToParquet

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
# @click.argument("input_path")
def preprocess_data(phase, master_url, input_path):
    spark = (
        SparkSession.builder.master(master_url).appName("book_impact_eda").getOrCreate()
    )
    df = Pipe(
        [
            GetYearFromDate("publishedDate"),
            GetMonthFromDate("publishedDate"),
            GetConcatenatedColumns(["title", "description"]),
            DropColumns(inputCols=COLS_TO_DROP),
            SaveToParquet(f"data/processed/{phase}/features"),
        ]
    ).transform(CsvDataFrame(input_path, spark))
    print(f"Saved {df.count()} rows of {phase} inputs")
    spark.stop()



if __name__ == "__main__":
    preprocess_data()
