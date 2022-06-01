from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import udf, col

import argparse
import pickle

class ALSCFModel:
	def __init__(self, obs_df, song_extra_info, spark, retrain = False):
		self.obs_df = obs_df.select(["user_index", "song_index", "obs"])
		self.obs_df = self.obs_df.\
			withColumn("user_index", self.obs_df["user_index"].cast(IntegerType())).\
			withColumn("song_index", self.obs_df["song_index"].cast(IntegerType())).\
			withColumn("obs", self.obs_df["obs"].cast(FloatType()))
		self.song_extra_info = song_extra_info
		self.spark = spark

		if retrain == True:
			als = ALS(maxIter=5, regParam=0.01, implicitPrefs=True, 
			userCol="user_index", itemCol="song_index", ratingCol="obs")
			model = als.fit(obs_df)
			model.save("file:///home/hadoop/Desktop/distributed_system/final_pj/model/als")

		self.model = ALSModel.load("file:///home/hadoop/Desktop/distributed_system/final_pj/model/als")

	def topNRecommend(self, user_index, N=20):
		user_subset = self.obs_df.where(self.obs_df.user_index == user_index)
		song_sim_df = self.spark.createDataFrame(
			self.model.recommendForUserSubset(user_subset, 20).collect()[0][1])
		return song_sim_df

	def recommend(self, user_index, N = 20):
		
		song_sim_df = self.topNRecommend(user_index, N)
		song_index_to_ID_file = open(
			"/home/hadoop/Desktop/distributed_system/final_pj/data/song_index_to_ID.pkl", "rb"
		)
		song_index_to_ID_dict = pickle.load(song_index_to_ID_file)
		map_index_to_ID = udf(lambda x : song_index_to_ID_dict.get(int(x)))
		song_sim_df = song_sim_df.withColumn("song_id", map_index_to_ID(col("song_index")))
		song_sim_df = song_sim_df.\
			join(song_extra_info, song_sim_df.song_id == song_extra_info.song_id).\
			select(["name", "rating"])
		song_sim_df = song_sim_df.orderBy(col("rating").desc())
		return song_sim_df

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--index', type = int, default = 42,
		help = "index of user for recommendation")
	parser.add_argument("--part", type = int, default = 200, 
		help = "spark sql shuffle partition number")
	args = parser.parse_args()


	spark = SparkSession.builder.master("local").\
		appName("user based CF model").\
		config("spark.driver.bindAddress","localhost").\
		config("spark.ui.port","4050").\
		getOrCreate()

	spark.conf.set("spark.sql.shuffle.partitions", args.part)

	obs_df = spark.read.option("header", True).\
		csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/implicit_sample")
	song_extra_info = spark.read.option("header", True).\
		csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/song_extra_info.csv")
	
	cfmodel = ALSCFModel(obs_df, song_extra_info, spark)
	recommendation = cfmodel.recommend(args.index)
	recommendation.show(30)

	a = input()
		
