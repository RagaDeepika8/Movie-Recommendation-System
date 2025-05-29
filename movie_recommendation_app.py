from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import streamlit as st
import os
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 8g pyspark-shell"

# Set Spark session with increased memory configs
spark = SparkSession.builder \
    .appName("MovieRecommendationSystem") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.storage.memoryFraction", "0.4") \
    .getOrCreate()

# Load Data (example)
@st.cache_resource
def load_data():
    movies = spark.read.csv(r"C:\\Users\\Lenovo\\Desktop\\movie_reommendation\\movie.csv", header=True, inferSchema=True)
    ratings = spark.read.csv(r"C:\\Users\\Lenovo\\Desktop\\movie_reommendation\\rating.csv", header=True, inferSchema=True)
    return movies, ratings


movies, ratings = load_data()

# Split data
train, validation, test = ratings.randomSplit([0.6, 0.2, 0.2], seed=42)

# Repartition to optimize shuffles
train = train.repartition(200)
validation = validation.repartition(100)

# ALS Grid Search function
def GridSearch(train, validation, num_iterations, reg_params, ranks):
    best_model = None
    best_rmse = float("inf")

    for rank in ranks:
        for reg in reg_params:
            als = ALS(
                maxIter=num_iterations,
                regParam=reg,
                rank=rank,
                userCol="userId",
                itemCol="movieId",
                ratingCol="rating",
                coldStartStrategy="drop",
                checkpointInterval=2,
                intermediateStorageLevel="MEMORY_AND_DISK"
            )
            model = als.fit(train)

            predictions = model.transform(validation)
            evaluator = RegressionEvaluator(
                metricName="rmse",
                labelCol="rating",
                predictionCol="prediction"
            )
            rmse = evaluator.evaluate(predictions)
            print(f"Rank: {rank}, RegParam: {reg}, RMSE: {rmse}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model

    return best_model

# Hyperparameters
num_iterations = 10
reg_params = [0.05, 0.1]
ranks = [8, 10, 12]

# Train final model with grid search
final_model = GridSearch(train, validation, num_iterations, reg_params, ranks)

# Evaluate on Test Set
predictions = final_model.transform(test)
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)
test_rmse = evaluator.evaluate(predictions)
st.write(f"🎥 Final Model Test RMSE: {test_rmse}")

# Stop Spark session cleanly at the end
spark.stop()

