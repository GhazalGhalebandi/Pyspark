# Databricks notebook source
# MAGIC %md
# MAGIC ### Predicting number of dengi cases 
# MAGIC  - changed the number of iterations

# COMMAND ----------

from pyspark import keyword_only
from pyspark.ml import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.ml.clustering import *
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import *
from pyspark.ml.param.shared import *
from pyspark.ml.param import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from HTMLParser import HTMLParser
from math import sqrt
from math import isnan
from datetime import datetime
import numpy
import re
import random

# COMMAND ----------

dbutils.fs.ls("s3a://sa-matching-production/data/dengai/")

# COMMAND ----------

dengue_features_train_schema = StructType([
  StructField('city', StringType(), True),
  StructField('year', FloatType(), True),
  StructField('weekofyear', FloatType(), True),
  StructField('week_start_date', DateType(), True),
  StructField('ndvi_ne', FloatType(), True),
  StructField('ndvi_nw', FloatType(), True),
  StructField('ndvi_se', FloatType(), True),
  StructField('ndvi_sw', FloatType(), True),
  StructField('precipitation_amt_mm', FloatType(), True),
  StructField('reanalysis_air_temp_k', FloatType(), True),
  StructField('reanalysis_avg_temp_k', FloatType(), True),
  StructField('reanalysis_dew_point_temp_k', FloatType(), True),
  StructField('reanalysis_max_air_temp_k', FloatType(), True),
  StructField('reanalysis_min_air_temp_k', FloatType(), True),
  StructField('reanalysis_precip_amt_kg_per_m2', FloatType(), True),
  StructField('reanalysis_relative_humidity_percent', FloatType(), True),
  StructField('reanalysis_sat_precip_amt_mm', FloatType(), True),
  StructField('reanalysis_specific_humidity_g_per_kg', FloatType(), True),
  StructField('reanalysis_tdtr_k', FloatType(), True),
  StructField('station_avg_temp_c', FloatType(), True),
  StructField('station_diur_temp_rng_c', FloatType(), True),
  StructField('station_max_temp_c', FloatType(), True),
  StructField('station_min_temp_c', FloatType(), True),
  StructField('station_precip_mm', FloatType(), True)
])

dengue_labels_train_schema = StructType([
  StructField('city', StringType(), True),
  StructField('year', FloatType(), True),
  StructField('weekofyear', FloatType(), True),
  StructField('total_cases', FloatType(), True)
])



# COMMAND ----------

# DBTITLE 1,load data
train = spark.read.schema(dengue_features_train_schema).csv("s3a://sa-matching-production/data/dengai/dengue_features_train.csv" , header = True)
label_dataset = spark.read.schema(dengue_labels_train_schema).csv('s3a://sa-matching-production/data/dengai/dengue_labels_train.csv', header = True)

# COMMAND ----------

train_labels = train.join(label_dataset , ['city' , 'year' , 'weekofyear'])

# COMMAND ----------

train_labels.select('city').distinct().show()

# COMMAND ----------

def numerical_cities(city):
  if city == 'iq':
    return '1'
  else:
    return '2'
  
transform_city = udf ( numerical_cities , StringType())

# COMMAND ----------

train_labels_transform_city = train_labels.withColumn('num_city', transform_city('city'))
train_labels_transform_city = train_labels_transform_city.drop('city')
display(train_labels_transform_city)

# COMMAND ----------

# DBTITLE 1,split date into year , month day column 
train_labels_transform_city_year = train_labels_transform_city.withColumn('year', substring(col('week_start_date'),0,4))
train_labels_transform_city_year_month = train_labels_transform_city_year.withColumn('month', substring(col('week_start_date'), 6,2))
train_labels_transform_city_year_month_day = train_labels_transform_city_year_month.withColumn('day', substring(col('week_start_date'), 9,2))

train_clean = train_labels_transform_city_year_month_day.drop('week_start_date')
display(train_clean)

# COMMAND ----------

# DBTITLE 1,fill nulls with mean value on each column 
def fill_with_mean(df, exclude=set()): 
    stats = df.agg(*(
        avg(c).alias(c) for c in df.columns if c not in exclude
    ))
    return df.na.fill(stats.first().asDict())

train_clean_nonull = fill_with_mean(train_clean, ["year", "month", "day" , "num_city"])

# COMMAND ----------

display(train_clean_nonull)

# COMMAND ----------

describe_mainTrain_df = train_clean_nonull.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalization 
# MAGIC   - x = value 
# MAGIC   - dl = min of attribute 
# MAGIC   - dh = max of attribute 
# MAGIC   - nl = min of expected range 
# MAGIC   - nh = max of expected range

# COMMAND ----------

mainTrain = train_clean_nonull

# COMMAND ----------

# call function
#normalize columns
def normalizing_column_1(c , dL, dH):
  nL = 0
  nH = 1
  numi = (float(c) - dL) * (nH-nL)
  denom = dH - dL
  div = float(numi) / float(denom)
  normalized = float(div + nL)
  return normalized

normalizing_column = udf(normalizing_column_1, DoubleType())


names = mainTrain.schema.names
for colname in names:
  dL = float(describe_mainTrain_df.collect()[3][colname])
  dH = float(describe_mainTrain_df.collect()[4][colname])
  mainTrain = mainTrain.withColumn('normalized_' + str(colname), 
                           normalizing_column(colname, lit(dL) , lit(dH))
                          )                                                                   
    

# COMMAND ----------

display(mainTrain)

# COMMAND ----------

# DBTITLE 1,write normalised mainTrain data into S3
mainTrain.write.mode('overwrite').parquet("s3a://sa-matching-production/dengi-ghazalg/normalized_mainTrain2_24NOV2017")


# COMMAND ----------

normalized_mainTrain = spark.read.parquet("s3a://sa-matching-production/dengi-ghazalg/normalized_mainTrain2_24NOV2017")
display(normalized_mainTrain)

# COMMAND ----------

normalized_mainTest = spark.read.parquet("s3a://sa-matching-production/dengi-ghazalg/normalized_mainTest2_24NOV2017")
display(normalized_mainTest)

# COMMAND ----------

normalized_mainTest.select('normalized_num_city').groupby('normalized_num_city').count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tensor Flow

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

normalized_mainTrain_pd = normalized_mainTrain.toPandas()
normalized_mainTest_pd = normalized_mainTest.toPandas()

# COMMAND ----------

normalized_mainTest_pd.year = normalized_mainTest_pd.year.astype(float)

# COMMAND ----------

non_feature_columns = ['year' , 'weekofyear']

# COMMAND ----------

feature_columns = [
  'normalized_year', 
  'normalized_weekofyear', 
  'normalized_ndvi_ne', 
  'normalized_ndvi_nw',
  'normalized_ndvi_se', 
  'normalized_ndvi_sw', 
  'normalized_precipitation_amt_mm',
  'normalized_reanalysis_air_temp_k', 
  'normalized_reanalysis_avg_temp_k',
  'normalized_reanalysis_dew_point_temp_k', 
  'normalized_reanalysis_max_air_temp_k',
  'normalized_reanalysis_min_air_temp_k',
  'normalized_reanalysis_precip_amt_kg_per_m2',
  'normalized_reanalysis_relative_humidity_percent',
  'normalized_reanalysis_sat_precip_amt_mm',
  'normalized_reanalysis_specific_humidity_g_per_kg', 
  'normalized_reanalysis_tdtr_k',
  'normalized_station_avg_temp_c', 
  'normalized_station_diur_temp_rng_c',
  'normalized_station_max_temp_c', 
  'normalized_station_min_temp_c',
  'normalized_station_precip_mm', 
  'normalized_num_city',
  'normalized_month',
  'normalized_day'
]

label_columns = ['total_cases']

# COMMAND ----------

import tensorflow as tf

tf.reset_default_graph()

feature_columns_tf = [
  tf.feature_column.numeric_column("normalized_year"), 
  tf.feature_column.numeric_column("normalized_weekofyear"), 
  tf.feature_column.numeric_column("normalized_ndvi_ne"), 
  tf.feature_column.numeric_column("normalized_ndvi_nw"),
  tf.feature_column.numeric_column("normalized_ndvi_se"), 
  tf.feature_column.numeric_column("normalized_ndvi_sw"), 
  tf.feature_column.numeric_column("normalized_precipitation_amt_mm"),
  tf.feature_column.numeric_column("normalized_reanalysis_air_temp_k"), 
  tf.feature_column.numeric_column("normalized_reanalysis_avg_temp_k"),
  tf.feature_column.numeric_column("normalized_reanalysis_dew_point_temp_k"), 
  tf.feature_column.numeric_column("normalized_reanalysis_max_air_temp_k"),
  tf.feature_column.numeric_column("normalized_reanalysis_min_air_temp_k"),
  tf.feature_column.numeric_column("normalized_reanalysis_precip_amt_kg_per_m2"),
  tf.feature_column.numeric_column("normalized_reanalysis_relative_humidity_percent"),
  tf.feature_column.numeric_column("normalized_reanalysis_sat_precip_amt_mm"),
  tf.feature_column.numeric_column("normalized_reanalysis_specific_humidity_g_per_kg"), 
  tf.feature_column.numeric_column("normalized_reanalysis_tdtr_k"),
  tf.feature_column.numeric_column("normalized_station_avg_temp_c"), 
  tf.feature_column.numeric_column("normalized_station_diur_temp_rng_c"),
  tf.feature_column.numeric_column("normalized_station_max_temp_c"), 
  tf.feature_column.numeric_column("normalized_station_min_temp_c"),
  tf.feature_column.numeric_column("normalized_station_precip_mm"), 
  tf.feature_column.numeric_column("normalized_num_city"),
  tf.feature_column.numeric_column("normalized_month"),
  tf.feature_column.numeric_column("normalized_day")
]

# Define the train inputs
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x = normalized_mainTrain_pd[feature_columns],
    y = normalized_mainTrain_pd[label_columns],
    num_epochs=None,
    shuffle=True)




# Define train inputs for evaluation
train_input_fn_eval = tf.estimator.inputs.pandas_input_fn(
    x = normalized_mainTrain_pd[feature_columns],
    y = normalized_mainTrain_pd[label_columns],
    num_epochs = 1,
    shuffle = False)


# Define test inputs for evaluation
test_input_fn_eval = tf.estimator.inputs.pandas_input_fn(
    x = normalized_mainTest_pd[feature_columns],
    y = None,
    num_epochs=1,
    shuffle=False)



classifier = tf.estimator.DNNRegressor(
  feature_columns = feature_columns_tf,
  hidden_units = [200, 300, 200, 300, 200, 300],
  optimizer = tf.train.ProximalAdagradOptimizer(
    learning_rate=0.01,
    l1_regularization_strength=0.001
  ),
  activation_fn = tf.nn.relu,
  model_dir= "/tmp/tf_dengai_mainTrain_ghazall_24Nov_v3"
)

# COMMAND ----------

# Train model.
classifier.train(input_fn = train_input_fn, steps=50000)

# COMMAND ----------

# Evaluate accuracy.
average_loss_train = classifier.evaluate(input_fn = train_input_fn_eval)["average_loss"]
print("Train Average Loss: {0:f}\n".format(average_loss_train))

# COMMAND ----------

# Evaluate accuracy.
average_loss_test = classifier.predict(input_fn = test_input_fn_eval)
#print("Test Average Loss: {0:f}\n".format(average_loss_test))

# COMMAND ----------

predictions_list = [prediction['predictions'][0] for prediction in average_loss_test]
len(predictions_list)

# COMMAND ----------

print predictions_list

# COMMAND ----------

list_to_df = pd.DataFrame({'predictions':predictions_list})
print (list_to_df)

# COMMAND ----------

cons = normalized_mainTest_pd[feature_columns].join(list_to_df)

# COMMAND ----------

cons.columns

# COMMAND ----------

spark.createDataFrame(
    cons
  ).count()

# COMMAND ----------

import math

# COMMAND ----------

def roundup(value):
  return math.ceil(value)

roundup = udf(roundup,FloatType())

def str_city(city):
  if(city == 1):
    return 'sj'
  elif(city == 2):
    return 'iq'

str_city = udf(str_city,StringType())

# COMMAND ----------

normalized_mainTest_pd.year = normalized_mainTest_pd.year.astype(float)

# COMMAND ----------

normalized_mainTest_pd

# COMMAND ----------

normalized_mainTest_pd_selected = normalized_mainTest_pd[['year', 'weekofyear']]


# COMMAND ----------

normalized_mainTest_pd_selected

# COMMAND ----------

result = pd.concat([cons, normalized_mainTest_pd_selected], axis=1)

# COMMAND ----------

result

# COMMAND ----------

pd_to_spark = spark.createDataFrame(result)
display(pd_to_spark)

# COMMAND ----------

new_df = pd_to_spark.select(
   col('normalized_num_city').alias('city'),  
   col('year'),
   col('weekofyear'),
   col('predictions').alias("total_cases")
)

display(new_df)

# COMMAND ----------

new_df.select('city').distinct().show()

# COMMAND ----------

def str_city(city):
  if(city == 1):
    return 'sj'
  elif(city == 0):
    return 'iq'

str_city = udf(str_city,StringType())

# COMMAND ----------

final_df3 = new_df.withColumn("city2", str_city('city')).drop('city')

# COMMAND ----------

display(final_df3)

# COMMAND ----------

submission_df12 = final_df3.select(
  col('city2').alias('city'),  
  col('year'),
  col('weekofyear'),
  col('total_cases')
)

# COMMAND ----------

display(submission_df12)

# COMMAND ----------

submission_df12.printSchema()

# COMMAND ----------

 import math

# COMMAND ----------

def roundup(value):
  return math.ceil(value)

roundup = udf(roundup,FloatType())
submission_df12 = submission_df12.withColumn('total_cases2', roundup('total_cases'))


# COMMAND ----------

display(submission_df12)

# COMMAND ----------

submission_df6 = submission_df12.drop('total_cases')
display(submission_df6)

# COMMAND ----------

submission_df7 = submission_df6.select(
  col('city'), 
  col('year'), 
  col('weekofyear'), 
  col('total_cases2').alias('total_cases')
)

# COMMAND ----------

display(submission_df7)

# COMMAND ----------

submission_df7.printSchema()

# COMMAND ----------


