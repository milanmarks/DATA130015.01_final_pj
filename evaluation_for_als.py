import pickle
import random

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, FloatType

from implicitFeedback.ALSModel import ALSCFModel

iter = 100

random.seed(42)

spark = SparkSession.builder.getOrCreate()

all_rate_df = spark.read.option("header", True).\
	csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/all_data_with_index").\
	select(["user_index", "song_index"])

all_rate_df = all_rate_df.\
	withColumn("user_index", all_rate_df["user_index"].cast(IntegerType())).\
	withColumn("song_index", all_rate_df["song_index"].cast(IntegerType()))

obs_df = spark.read.option("header", True).\
	csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/implicit_sample")

obs_df = obs_df.\
	withColumn("user_index", obs_df["user_index"].cast(IntegerType())).\
	withColumn("song_index", obs_df["song_index"].cast(IntegerType())).\
	withColumn("obs", obs_df["obs"].cast(FloatType()))

song_extra_info = spark.read.option("header", True).\
	csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/song_extra_info.csv")

user_index_to_ID_file = open(
			"/home/hadoop/Desktop/distributed_system/final_pj/data/user_index_to_ID.pkl", "rb"
		)
user_index_to_ID_dict = pickle.load(user_index_to_ID_file)
user_index_to_ID_file.close()

song_index_to_ID_file = open(
			"/home/hadoop/Desktop/distributed_system/final_pj/data/song_index_to_ID.pkl", "rb"
		)
song_index_to_ID_dict = pickle.load(song_index_to_ID_file)
song_index_to_ID_file.close()

# random select 100 user for evaluation
user_for_eval = [random.choice(list(user_index_to_ID_dict)) for i in range(iter)]

# filter the user-song df from all_rate_df
song_for_eval_df = all_rate_df.filter(all_rate_df.user_index.isin(user_for_eval))

def precision(user_index, recommend_df):
	user_listen_df = song_for_eval_df.filter(song_for_eval_df.user_index == user_index).\
		select("song_index")
	recommend_df = recommend_df.\
		join(user_listen_df, recommend_df.song_index == user_listen_df.song_index, "left").\
		toDF("song_index", "rating", "song_appear")

	recommend_df = recommend_df.\
		withColumn("indicator", F.when(F.col("song_appear").isNull(), F.lit(0)).otherwise(F.lit(1)))

	recommend_df = recommend_df.select("indicator")
	precision = recommend_df.groupBy().sum().collect()[0][0] / recommend_df.count()
	return precision
	
cfmodel = ALSCFModel(obs_df, song_extra_info, spark)
precision_list = [0 for _ in range(iter)]

for i in range(iter):
	try:
		precision_list[i] = precision(user_for_eval[i], cfmodel.topNRecommend(user_for_eval[i]))
	except:
		precision_list[i] = 0

print(precision_list)
print(sum(precision_list) / len(precision_list))




	
