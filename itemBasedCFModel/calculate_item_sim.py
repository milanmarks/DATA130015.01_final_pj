from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType, col, row_number
from pyspark.sql.types import IntegerType, FloatType
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from pyspark.ml.feature import Normalizer
from pyspark.sql.window import Window


from numpy import linalg as LA

import argparse


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('part', type = int)
	args = parser.parse_args()

	spark = SparkSession.builder.master("local").\
		appName("calculate similarity").\
		config("spark.driver.bindAddress","localhost").\
		config("spark.ui.port","4050").\
		getOrCreate()

	spark.conf.set("spark.sql.shuffle.partitions", args.part)


	df = spark.read.option("header", True).\
		csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/data_with_index").\
		select(["user_index", "song_index", "obs"])

	# turn column from StringType to IntegerType and Floatype
	df = df.withColumn("user_index", df["user_index"].cast(IntegerType())).\
		withColumn("song_index", df["song_index"].cast(IntegerType())).\
		withColumn("obs", df["obs"].cast(FloatType()))

	@pandas_udf("user_index int, song_index int, obs float", PandasUDFType.GROUPED_MAP)
	def L2normalize(pdf):
		obs = pdf.obs
		return pdf.assign(obs=obs / LA.norm(obs))

	# L2 normalize for same song
	df = df.groupBy("song_index").apply(L2normalize)

	# calculate similarity by matrix multiplication
	rate_df = df.alias("df1").\
		join(df.alias("df2"), col("df1.user_index") == col("df2.user_index")).\
		groupBy(col("df1.song_index"), col("df2.song_index")).\
		agg(F.sum(col("df1.obs") * col("df2.obs"))).\
		toDF("song1", "song2", "similarity")


	windowDept = Window.partitionBy("song1").orderBy(col("similarity").desc())
	rate_df = rate_df.withColumn("row", row_number().over(windowDept)).\
		filter(col("row") <= 20)
	
	rate_df.show()
	a=input()
	"""
	rate_df.repartition(1).write.option("header", True).mode("overwrite").\
		csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/song_similarity")
	"""
