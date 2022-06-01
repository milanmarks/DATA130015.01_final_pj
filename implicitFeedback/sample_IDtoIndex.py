from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType

spark = SparkSession.builder.getOrCreate()

df = spark.read.option("header", True).\
		csv('file:///home/hadoop/Desktop/distributed_system/final_pj/data/train.csv').\
		select(["msno", "song_id", "target"])

df = df.withColumn("obs", df.target + 1).drop("target")

data_index = spark.read.option("header", True).\
    csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/data_with_index").\
	drop("obs")

df = df.join(data_index, ["msno", "song_id"])

df.write.option("header", True).mode("overwrite").\
	csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/implicit_sample")
