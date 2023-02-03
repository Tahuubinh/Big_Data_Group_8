from pyspark.sql.functions import col, explode
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from operator import add
#import sys,os
import config
#import tempfile
import json

if __name__ == "__main__":
    APP_NAME="PreprocessData"
    print("start")
    app_config = config.Config(elasticsearch_host="elasticsearch",
                               elasticsearch_port="9200",
                               elasticsearch_input_json="yes",
                               elasticsearch_nodes_wan_only="true",
                               hdfs_namenode="hdfs://namenode:9000"
                               )
    # print("init spark session")
    spark = app_config.initialize_spark_session(APP_NAME)
    # print("read file csv")

    sc = SparkContext
    # sc.setCheckpointDir('checkpoint')
    # spark = SparkSession.builder.appName('Recommendations').getOrCreate()

    movies = spark.read.csv("hdfs://namenode:9000/data/movies.csv",header=True)
    ratings = spark.read.csv("hdfs://namenode:9000/data/ratings.csv",header=True)

    # ratings.show()
    # ratings.printSchema()

    ratings = ratings.\
        withColumn('userId', col('userId').cast('integer')).\
        withColumn('movieId', col('movieId').cast('integer')).\
        withColumn('rating', col('rating').cast('float')).\
        drop('timestamp')
    # ratings.show()



    # Count the total number of ratings in the dataset
    numerator = ratings.select("rating").count()

    # Count the number of distinct userIds and distinct movieIds
    num_users = ratings.select("userId").distinct().count()
    num_movies = ratings.select("movieId").distinct().count()

    # Set the denominator equal to the number of users multiplied by the number of movies
    denominator = num_users * num_movies

    # Divide the numerator by the denominator
    sparsity = (1.0 - (numerator *1.0)/denominator)*100
    # print("The ratings dataframe is ", "%.2f" % sparsity + "% empty.")

    # Group data by userId, count ratings
    userId_ratings = ratings.groupBy("userId").count().orderBy('count', ascending=False)
    # userId_ratings.show()

    # Group data by userId, count ratings
    movieId_ratings = ratings.groupBy("movieId").count().orderBy('count', ascending=False)

    # Create test and train set
    (train, test) = ratings.randomSplit([0.8, 0.2], seed = 1234)

    # Create ALS model
    als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative = True, implicitPrefs = False, coldStartStrategy="drop")

    # Confirm that a model called "als" was created
    print(type(als))

    # Add hyperparameters and their respective values to param_grid
    # param_grid = ParamGridBuilder() \
    #             .addGrid(als.rank, [10, 50, 100, 150]) \
    #             .addGrid(als.regParam, [.01, .05, .1, .15]) \
    #             .build()
                #             .addGrid(als.maxIter, [5, 50, 100, 200]) \

    param_grid = ParamGridBuilder() \
                .addGrid(als.rank, [10]) \
                .addGrid(als.regParam, [.01]) \
                .build()
                #             .addGrid(als.maxIter, [5, 50, 100, 200]) \

            
    # Define evaluator as RMSE and print length of evaluator
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction") 
    print ("Num models to be tested: ", len(param_grid))



    # Build cross validation using CrossValidator
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

    # Confirm cv was built
    print(cv)

    #Fit cross validator to the 'train' dataset
    model = cv.fit(train)

    # Save model
    # print(model.getNumFolds())
    # print(model.avgMetrics[0])
    # path = tempfile.mkdtemp()
    # print(path)
    # model_path = path + "/model"
    # model.write().save(model_path)
    
    # model.write().overwrite().save('/model/')
    # cvModelRead = CrossValidatorModel.read().load('/model/')

    #Extract best model from the cv model above
    best_model = model.bestModel
    best_model.save('hdfs://namenode:9000/data/ALSmodel')
    best_model = ALSModel.load('hdfs://namenode:9000/data/ALSmodel')

    # Print best_model
    print(type(best_model))

    txt_content = dict()
    # Complete the code below to extract the ALS model parameters
    print("**Best Model**")

    # # Print "Rank"
    print("  Rank:", best_model._java_obj.parent().getRank())
    txt_content['rank'] = best_model._java_obj.parent().getRank()

    # Print "MaxIter"
    print("  MaxIter:", best_model._java_obj.parent().getMaxIter())
    txt_content['maxiter'] = best_model._java_obj.parent().getMaxIter()

    # Print "RegParam"
    print("  RegParam:", best_model._java_obj.parent().getRegParam())
    txt_content['regparam'] = best_model._java_obj.parent().getRegParam()

    json_object = json.dumps(txt_content, indent=4)
    with open("/result/model.json", "w") as outfile:
        outfile.write(json_object)







