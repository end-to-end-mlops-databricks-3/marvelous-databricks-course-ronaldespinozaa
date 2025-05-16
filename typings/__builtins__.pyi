
from databricks.sdk.runtime import dbutils
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import udf as U
from pyspark.sql.context import SQLContext
from typing import Any

udf = U
spark: SparkSession
sc = spark.sparkContext
sqlContext: SQLContext
sql = sqlContext.sql
table = sqlContext.table
getArgument = dbutils.widgets.getArgument

def displayHTML(html: str) -> None:
    pass  # Indica que la función no hace nada por el momento

def display(input: Any = None, *args: Any, **kwargs: Any) -> None:
    pass  # Indica que la función no hace nada por el momento
