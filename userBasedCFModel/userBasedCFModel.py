from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType, col, udf
from pyspark.sql.types import IntegerType, FloatType

import argparse
import pickle

class UserBasedCFModel():
	"""
	Implementation of user-based Collabrative Filtering Model
	"""
	def __init__(self, rate_df, sim_df, song_extra_info):
		"""
		Args:
			rate_df: the user listening history dataframe
        	sim_df: the similarity dataframe
			song_extra_info: dataframe of song_id and song name
		"""
		self.rate_df = rate_df.select(["user_index", "song_index"])
		self.rate_df = self.rate_df.\
			withColumn("user_index", self.rate_df["user_index"].cast(IntegerType())).\
			withColumn("song_index", self.rate_df["song_index"].cast(IntegerType()))
		self.sim_df = sim_df
		self.song_extra_info = song_extra_info.select(["song_id", "name"])

	def findKNN(self, user_index, k = 20):
		"""
		find k nearest neighbor of user
		[Suspended] In this version, we calculate kNN in advance to save time
 
        Args:
            user_index: the index of user we care about
            k: the number of the nearest neighbor

		Returns:
			A dataframe containing kNN
		"""
		neighbor_df = self.sim_df.filter(self.sim_df.user1 == user_index).\
				drop("user1")
		neighbor_df = neighbor_df.filter(neighbor_df.user2 != user_index).\
                orderBy(col("similarity").desc()).\
				limit(k)
	
		return neighbor_df

	def topNRecommend(self, user_index, N = 20, k = 20):
		"""
		Args:
			user_index: the user waiting for recommendation
			N: number of sone recommendation
			k: number of neighbor

		Returns:
			A list of index of songs
		"""

		# get the user-neigbhor-similarity dataframe
		neighbor_sim_df = self.sim_df.filter(self.sim_df.user1 == user_index)
		
		# get the neighbor's song from rate_df
		song_sim_df = neighbor_sim_df.join(self.rate_df, neighbor_sim_df.user2 == self.rate_df.user_index)

		# get the song list user listened
		user_listen_df = self.rate_df.filter(self.rate_df.user_index == user_index)
		
		# exclude the user-listened-song
		song_sim_df = song_sim_df.join(
			user_listen_df, 
			song_sim_df.song_index == user_listen_df.song_index,
			"leftanti"
		)

		# get the song-similarity dataframe
		song_sim_df = song_sim_df.\
			groupby("song_index").\
			agg(F.sum("similarity")).\
			toDF("song_index", "sumOfSimilarity").\
			orderBy(col("sumOfSimilarity").desc()).\
			limit(N)

		return song_sim_df

	def recommend(self, user_index, N = 20, k = 20):
		"""
		Args:
			user_index: the user waiting for recommendation
			N: number of sone recommendation
			k: number of neighbor

		Returns:
			A list of names of songs
		"""
		song_index_to_ID_file = open(
			"/home/hadoop/Desktop/distributed_system/final_pj/data/song_index_to_ID.pkl", "rb"
		)
		song_index_to_ID_dict = pickle.load(song_index_to_ID_file)

		map_index_to_ID = udf(lambda x : song_index_to_ID_dict.get(int(x)))

		
		song_sim_df = self.topNRecommend(user_index, N, k)
		song_sim_df = song_sim_df.withColumn("song_id", map_index_to_ID(col("song_index")))

		song_sim_df = song_sim_df.\
			join(self.song_extra_info, song_sim_df.song_id == self.song_extra_info.song_id).\
			select(["name", "sumOfSimilarity"])

		song_sim_df = song_sim_df.orderBy(col("sumOfSimilarity").desc())

		return song_sim_df
		

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--index', type = int,
		help = "index of user for recommendation")
	parser.add_argument("--part", type = int, help = "spark sql shuffle partition number")
	args = parser.parse_args()


	spark = SparkSession.builder.master("local").\
		appName("user based CF model").\
		config("spark.driver.bindAddress","localhost").\
		config("spark.ui.port","4050").\
		getOrCreate()

	spark.conf.set("spark.sql.shuffle.partitions", args.part)

	rate_df = spark.read.option("header", True).\
		csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/data_with_index")
	sim_df = spark.read.option("header", True).\
		csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/user_similarity")
	song_extra_info = spark.read.option("header", True).\
		csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/song_extra_info.csv")

	
	cfmodel = UserBasedCFModel(rate_df, sim_df, song_extra_info)
	recommendation = cfmodel.recommend(args.index)
	recommendation.show(30)

	a = input()
	
		
		
		
		
		
		
		
		
		
        

