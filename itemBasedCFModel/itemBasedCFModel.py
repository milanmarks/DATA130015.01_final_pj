from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType, col, udf
from pyspark.sql.types import IntegerType, FloatType

import argparse
import pickle

class ItemBasedCFModel():
	"""
	Implement the item-based Collabrative Filtering Model
	"""
	
	def __init__(self, rate_df, sim_df, song_extra_info):
		"""
		Args:
			rate_df: the user-song-rate dataframe
			sim_df: the song-song-similarity dataframe
			song_extra_info: the id-name dataframe
		"""
		self.rate_df = rate_df.select(["user_index", "song_index"])
		self.rate_df = self.rate_df.\
			withColumn("user_index", self.rate_df["user_index"].cast(IntegerType())).\
			withColumn("song_index", self.rate_df["song_index"].cast(IntegerType()))

		self.sim_df = sim_df
		self.song_extra_info = song_extra_info.select(["song_id", "name"])

	def topNRecommend(self, user_index, N = 20):
		# get the user-song df
		user_listen_df = self.rate_df.filter(self.rate_df.user_index == user_index)
		
		# join with song-neighbor-sim df
		song_sim_df = user_listen_df.join(
			self.sim_df, user_listen_df.song_index == self.sim_df.song1
			).\
			groupBy("song2").\
			agg(F.sum("similarity")).\
			toDF("song_index", "sumOfSimilarity")

		# delete those songs listened by user
		song_sim_df = song_sim_df.join(
			user_listen_df,\
			song_sim_df.song_index == user_listen_df.song_index,
			"leftanti"
		)

		song_sim_df = song_sim_df.\
			orderBy(col("sumOfSimilarity").desc()).\
			limit(N)

		return song_sim_df


	def recommend(self, user_index, N = 20):
		song_index_to_ID_file = open(
			"/home/hadoop/Desktop/distributed_system/final_pj/data/song_index_to_ID.pkl", "rb"
		)
		song_index_to_ID_dict = pickle.load(song_index_to_ID_file)

		map_index_to_ID = udf(lambda x : song_index_to_ID_dict.get(int(x)))

		
		song_sim_df = self.topNRecommend(user_index, N)
		song_sim_df = song_sim_df.withColumn("song_id", map_index_to_ID(col("song_index")))

		song_sim_df = song_sim_df.\
			join(self.song_extra_info, song_sim_df.song_id == self.song_extra_info.song_id).\
			select(["name", "sumOfSimilarity"])

		song_sim_df = song_sim_df.orderBy(col("sumOfSimilarity").desc())
			

		return song_sim_df


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('user_index', metavar='user', type = int,
		help = "index of user for recommendation")
	args = parser.parse_args()

	spark = SparkSession.builder.master("local").\
		appName("user based CF model").\
		config("spark.driver.bindAddress","localhost").\
		config("spark.ui.port","4050").\
		getOrCreate()

	rate_df = spark.read.option("header", True).\
		csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/data_with_index")
	sim_df = spark.read.option("header", True).\
		csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/song_similarity")
	song_extra_info = spark.read.option("header", True).\
		csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/song_extra_info.csv")

	
	cfmodel = ItemBasedCFModel(rate_df, sim_df, song_extra_info)
	recommendation = cfmodel.recommend(args.user_index)
	recommendation.show(30)

	a = input()
	
			
			
