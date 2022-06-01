import pyspark as ps
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.ml.feature import StringIndexer

import pickle

if __name__ == "__main__":
	spark = SparkSession.builder.getOrCreate()
	# read csv file, get sample data, add column with value one
	df = spark.read.option("header", True).\
		csv('file:///home/hadoop/Desktop/distributed_system/final_pj/data/train.csv').\
		select(["msno", "song_id"])
	
	# turn string id to integer index
	stringIndexer1 = StringIndexer(inputCol = "msno", outputCol = "user_index")
	stringIndexer_model1 = stringIndexer1.fit(df)
	stringIndexer2 = StringIndexer(inputCol = "song_id", outputCol = "song_index")
	stringIndexer_model2 = stringIndexer2.fit(df)
	df = stringIndexer_model2.transform(stringIndexer_model1.transform(df))

	df = df.withColumn("obs", lit(1))

	df.write.option("header", True).mode("overwrite").\
		csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/all_data_with_index")

	df = df.sample(0.1, seed = 42)
	df.write.option("header", True).mode("overwrite").\
		csv("file:///home/hadoop/Desktop/distributed_system/final_pj/data/data_with_index")
	
	# get the dictionary map index to true id
	meta1 = [
		f.metadata for f in df.schema.fields if f.name == "user_index"
		]

	user_index_to_ID_dict = dict(enumerate(meta1[0]["ml_attr"]["vals"]))
	meta2 = [
		f.metadata for f in df.schema.fields if f.name == "song_index"
		]
	song_index_to_ID_dict = dict(enumerate(meta2[0]["ml_attr"]["vals"]))

	user_index_to_ID_file = open(
		"/home/hadoop/Desktop/distributed_system/final_pj/data/user_index_to_ID.pkl", 
		"wb"
	)
	pickle.dump(user_index_to_ID_dict, user_index_to_ID_file)
	user_index_to_ID_file.close()

	song_index_to_ID_file = open(
		"/home/hadoop/Desktop/distributed_system/final_pj/data/song_index_to_ID.pkl", 
		"wb"
	)
	pickle.dump(song_index_to_ID_dict, song_index_to_ID_file)
	song_index_to_ID_file.close()
	
	
	
	
	
    
