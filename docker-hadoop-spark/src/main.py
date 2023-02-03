# coding=utf-8
import pyspark
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from operator import add
import sys,os
from pyspark.sql.types import *

import config


if __name__ == "__main__":
    
    APP_NAME="PreprocessData"
    print("start")
    print("------------------------------------------------------------------------------------------------------")

    app_config = config.Config(elasticsearch_host="elasticsearch",
                               elasticsearch_port="9200",
                               elasticsearch_input_json="yes",
                               elasticsearch_nodes_wan_only="true",
                               hdfs_namenode="hdfs://namenode:9000"
                               )
    print("init spark session")
    print("------------------------------------------------------------------------------------------------------")
    spark = app_config.initialize_spark_session(APP_NAME)
    print("read file csv")
    print("------------------------------------------------------------------------------------------------------")
    # sc = spark.sparkContext
    # sc.addPyFile(os.path.dirname(__file__)+"/patterns.py")
    df_user = spark.read.csv("hdfs://namenode:9000/data/users/user.csv")
    print("read user done!")
    print("------------------------------------------------------------------------------------------------------")
    df_user_0 = spark.read.csv("hdfs://namenode:9000/data/users/0.csv")
    
    def count_episode_watched():
        sql = spark.sql("Select u.username, count(u_0.watched_episodes) as total_episode_watched\
                         From user u, 0 u_0\
                         Where u.user_id = 0")
        
        sql.show()
    
    count_episode_watched()
    # raw_recruit_df = spark.read.schema(schema).option("multiline","true").json("hdfs://namenode:9000/data/rawdata/*.json")
    # # raw_recruit_df.show(5)
    # extracted_recruit_df=raw_recruit_df.select(raw_recruit_df["name"].alias("CompanyName"),
    #       udfs.extract_framework_plattform("Mô tả công việc","Yêu cầu ứng viên").alias("FrameworkPlattforms"),
    #       udfs.extract_language("Mô tả công việc","Yêu cầu ứng viên").alias("Languages"),
    #       udfs.extract_design_pattern("Mô tả công việc","Yêu cầu ứng viên").alias("DesignPatterns"),
    #       udfs.extract_knowledge("Mô tả công việc","Yêu cầu ứng viên").alias("Knowledges"),
    #       udfs.normalize_salary("Quyền lợi").alias("Salaries")
    #       )
    # extracted_recruit_df.cache()
    # # extracted_recruit_df.show(5)

    # ##========save extracted_recruit_df to hdfs========================
    # df_to_hdfs=(extracted_recruit_df,)
    # df_hdfs_name = ("extracted_recruit.json",)
    # io_cluster.save_dataframes_to_hdfs("data/extracteddata", app_config, df_to_hdfs, df_hdfs_name)

    # ##========make some query==========================================
    # knowledge_df = queries.get_counted_knowledge(extracted_recruit_df)
    # knowledge_df.cache()
    # # knowledge_df.show(5)

    # udfs.broadcast_labeled_knowledges(sc,patterns.labeled_knowledges)
    # grouped_knowledge_df = queries.get_grouped_knowledge(knowledge_df)
    # grouped_knowledge_df.cache()
    # # grouped_knowledge_df.show()

    # extracted_recruit_df = extracted_recruit_df.drop("Knowledges")
    # extracted_recruit_df.cache()

    # ##========save some df to elasticsearch========================
    # df_to_elasticsearch=(
    #                      extracted_recruit_df,
    #                     #  knowledge_df,
    #                      grouped_knowledge_df
    #                      )
    
    # df_es_indices = (
    #                  "recruit",
    #               #    "knowledges",
    #                  "grouped_knowledges"
    #                  )
    # io_cluster.save_dataframes_to_elasticsearch(df_to_elasticsearch,df_es_indices,app_config.get_elasticsearch_conf())