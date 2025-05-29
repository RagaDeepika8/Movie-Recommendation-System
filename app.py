
# app.py
import streamlit as st
import time
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pow

# Initialize SparkSession
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("MovieRecommendation") \
    .config("spark.driver.memory", "10g") \
    .getOrCreate()


st.title('Movie Recommendation System')

st.write("Loading datasets...")
# Load datasets (update the paths if needed)
movies = spark.read.load("movie.csv", format='csv', header=True)
ratings = spark.read.load("rating.csv", format='csv', header=True)
links = spark.read.load("link.csv", format='csv', header=True)
tags = spark.read.load("tag.csv", format='csv', header=True)

if st.checkbox("Show Ratings Dataset"):
    st.dataframe(ratings.sample(False, 0.01, seed=42).toPandas())


# Cast columns to appropriate types
df = ratings.withColumn('userId', ratings['userId'].cast('int')) \
            .withColumn('movieId', ratings['movieId'].cast('int')) \
            .withColumn('rating', ratings['rating'].cast('float'))

# Show schema
if st.checkbox("Show Data Schema"):
    df.printSchema()

# Split the data
train, validation, test = df.randomSplit([0.6, 0.2, 0.2], seed=0)
st.write("The number of ratings in each set:")
st.write(f"Train: {train.count()}, Validation: {validation.count()}, Test: {test.count()}")

# Define RMSE calculation
def RMSE(predictions):
    squared_diff = predictions.withColumn("squared_diff", pow(col("rating") - col("prediction"), 2))
    mse = squared_diff.selectExpr("mean(squared_diff) as mse").first().mse
    return mse ** 0.5

# Perform Grid Search for ALS
from pyspark.ml.recommendation import ALS

def GridSearch(train, valid, num_iterations, reg_param, n_factors):
    min_rmse = float('inf')
    best_n = -1
    best_reg = 0
    best_model = None

    for n in n_factors:
        for reg in reg_param:
            als = ALS(rank=n,
                      maxIter=num_iterations,
                      seed=0,
                      regParam=reg,
                      userCol="userId",
                      itemCol="movieId",
                      ratingCol="rating",
                      coldStartStrategy="drop")
            model = als.fit(train)
            predictions = model.transform(valid)
            rmse = RMSE(predictions)
            st.write(f"{n} latent factors and regularization {reg}: validation RMSE = {rmse:.4f}")

            if rmse < min_rmse:
                min_rmse = rmse
                best_n = n
                best_reg = reg
                best_model = model

    pred = best_model.transform(train)
    train_rmse = RMSE(pred)

    st.success(f"Best Model: {best_n} latent factors, regularization = {best_reg}")
    st.info(f"Training RMSE: {train_rmse:.4f} | Validation RMSE: {min_rmse:.4f}")
    
    return best_model

# Hyperparameters
num_iterations = 10
ranks = [6, 8, 10, 12]
reg_params = [0.05, 0.1, 0.2, 0.4, 0.8]

if st.button("Train Model with Grid Search"):
    with st.spinner("Training... Please wait."):
        final_model = GridSearch(train, validation, num_iterations, reg_params, ranks)
        pred_test = final_model.transform(test)
        st.success(f"Testing RMSE: {RMSE(pred_test):.4f}")

        user_id = st.number_input('Enter a User ID to Recommend Movies', min_value=1, value=25)
        
        single_user = test.filter(test['userId'] == user_id).select(['movieId', 'userId'])
        
        if single_user.count() == 0:
            st.warning("User not found in the test set.")
        else:
            st.subheader(f"Movies rated by User {user_id}")
            st.dataframe(single_user.toPandas())
