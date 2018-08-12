# Databricks notebook source
# MAGIC %md ## Apache Spark MLLib

# COMMAND ----------

# MAGIC %md ### Part One - Load and Prepare the data 

# COMMAND ----------

cleanedTaxes = sqlContext.sql("SELECT * FROM cleaned_taxes")
cleanedTaxes.show()

# COMMAND ----------

# taxes2013 = spark.read\
#   .option("header", "true")\
#   .csv("dbfs:/databricks-datasets/data.gov/irs_zip_code_data/data-001/2013_soi_zipcode_agi.csv")
# taxes2013.createOrReplaceTempView("taxes2013")

# COMMAND ----------

# %sql
# DROP TABLE IF EXISTS cleaned_taxes;

# CREATE TABLE cleaned_taxes AS
# SELECT 
#   state, 
#   int(zipcode / 10) as zipcode,
#   int(mars1) as single_returns,
#   int(mars2) as joint_returns,
#   int(numdep) as numdep,
#   double(A02650) as total_income_amount,
#   double(A00300) as taxable_interest_amount,
#   double(a01000) as net_capital_gains,
#   double(a00900) as biz_net_income
# FROM taxes2013

# COMMAND ----------

markets = spark.read\
  .option("header", "true")\
  .csv("dbfs:/databricks-datasets/data.gov/farmers_markets_geographic_data/data-001/market_data.csv")

# COMMAND ----------

summedTaxes = cleanedTaxes\
  .groupBy("zipcode")\
  .sum()
  
summedTaxes.show()

# COMMAND ----------

## getting zipcode buckets
cleanedMarkets = markets\
  .selectExpr("*", "int(zip / 10) as zipcode")\
  .groupBy("zipcode")\
  .count()\
  .selectExpr("double(count) as count", "zipcode as zip")

cleanedMarkets.show()

# COMMAND ----------

# MAGIC %md Join the two cleaned datasets into one dataset for analysis.
# MAGIC 
# MAGIC * Outer join `cleanedMarkets` to `summedTaxes` using `zip` and `zipcode` as the join variable.
# MAGIC * Name the resulting dataset `joined`.

# COMMAND ----------

joined = cleanedMarkets\
  .join(summedTaxes, cleanedMarkets["zip"] == summedTaxes["zipcode"], "outer")

# COMMAND ----------

display(joined)

# COMMAND ----------

prepped = joined.na.fill(0)
display(prepped)

# COMMAND ----------

# MAGIC %md ### Part Two -Use MLLib with Spark
# MAGIC * Put all the features into a single vector.  
# MAGIC * Create an array to list the names of all the **non-feature** columns: `zip`, `zipcode`, `count`, call it `nonFeatureCols`.
# MAGIC * Create a list of names called `featureCols` which excludes the columns in `nonFeatureCols`.

# COMMAND ----------

nonFeatureCols = {'zip', 'zipcode', 'count'}
featureCols = [column for column in prepped.columns if column not in nonFeatureCols]
print(featureCols)

# COMMAND ----------

# MAGIC %md * Use the `VectorAssembler` from `pyspark.ml.feature` to add a `features` vector to the `prepped` dataset.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=[column for column in featureCols],
    outputCol='features')

## new dataset
finalPrep = assembler.transform(prepped)
display(finalPrep.select('zipcode', 'features'))

# COMMAND ----------

display(finalPrep.drop("zip").drop("zipcode").drop("features"))

# COMMAND ----------

(training, test) = finalPrep.randomSplit((0.7, 0.3))

training.cache()
test.cache()

print(training.count())
print(test.count())

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

## make the model object
lrModel = LinearRegression()\
  .setLabelCol("count")\
  .setFeaturesCol("features")\
  .setElasticNetParam(0.5)

print("Printing out the model Parameters:")
print("-"*20)
print(lrModel.explainParams())
print("-"*20)

# COMMAND ----------

lrFitted = lrModel.fit(training)

# COMMAND ----------

# MAGIC %md 
# MAGIC * Make a prediction by using the `transform` method on `lrFitted`, passing it the `test` dataset. 
# MAGIC * Store the results in `holdout`.
# MAGIC * `transform` adds a new column called "prediction" to the data we passed into it.

# COMMAND ----------

holdout = lrFitted.transform(test)
display(holdout.select("prediction", "count"))

# COMMAND ----------

## this will tell us how many we got correctly
holdout = holdout.selectExpr(\
                             "prediction as raw_prediction", \
                             "double(round(prediction)) as prediction", \
                             "count", \
                             """CASE double(round(prediction)) = count 
                                WHEN true then 1
                                ELSE 0
                                END as equal""")
display(holdout)

# COMMAND ----------

# MAGIC %md * Use another `selectExpr` to `display` the proportion of predictions that were exactly correct.

# COMMAND ----------

# give us our accuracy, but the score is not good
display(holdout.selectExpr("sum(equal)/sum(1)"))

# COMMAND ----------

# MAGIC %md * Use `RegressionMetrics` to get more insight into the model performance. NOTE: Regression metrics requires input formatted as tuples of `double`s where the first item is the `prediction` and the second item is the observation (in this case the observation is `count`). 

# COMMAND ----------

# get a more accuracte model using RegressionMetrics and new methods
from pyspark.mllib.evaluation import RegressionMetrics

mapped = holdout.select("prediction", "count").rdd.map(lambda x: (float(x[0]), float(x[1])))
rm = RegressionMetrics(mapped)

print ("MSE: ", rm.meanSquaredError)
print ("MAE: ", rm.meanAbsoluteError)
print ("RMSE Squared: ", rm.rootMeanSquaredError)
print ("R Squared: ", rm.r2)
print ("Explained Variance: ", rm.explainedVariance)

# COMMAND ----------

# MAGIC %md Using a RandomForestRegressor and building a param grid for tuning the hyperparameters.
# MAGIC * Make a piepline to feed the algorithm into a CrossValidator to help prevent overfitting

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

rfModel = RandomForestRegressor()\
  .setLabelCol("count")\
  .setFeaturesCol("features")
  
paramGrid = ParamGridBuilder()\
  .addGrid(rfModel.maxDepth, [5, 10])\
  .addGrid(rfModel.numTrees, [20, 60])\
  .build()

steps = [rfModel]

pipeline = Pipeline().setStages(steps)

cv = CrossValidator()\
  .setEstimator(pipeline)\
  .setEstimatorParamMaps(paramGrid)\
  .setEvaluator(RegressionEvaluator().setLabelCol("count"))

pipelineFitted = cv.fit(training)

# COMMAND ----------

## looking for the "best" model of the params
print("The Best Parameters:\n--------------------")
print(pipelineFitted.bestModel.stages[0])

# COMMAND ----------

holdout2 = pipelineFitted.bestModel\
  .transform(test)\
  .selectExpr("prediction as raw_prediction", \
    "double(round(prediction)) as prediction", \
    "count", \
    """CASE double(round(prediction)) = count 
  WHEN true then 1
  ELSE 0
END as equal""")
  
display(holdout2)

# COMMAND ----------

# MAGIC %md * Show the `RegressionMetrics` for the new model results.

# COMMAND ----------

from pyspark.mllib.evaluation import RegressionMetrics

mapped2 = holdout2.select("prediction", "count").rdd.map(lambda x: (float(x[0]), float(x[1])))
rm2 = RegressionMetrics(mapped2)

print ("MSE: ", rm2.meanSquaredError)
print ("MAE: ", rm2.meanAbsoluteError)
print ("RMSE Squared: ", rm2.rootMeanSquaredError)
print ("R Squared: ", rm2.r2)
print ("Explained Variance: ", rm2.explainedVariance)

# COMMAND ----------

# MAGIC %md * See if there an improvement in the "exactly right" proportion.  And as expected there is a significant improvement over the first model

# COMMAND ----------

display(holdout2.selectExpr("sum(equal)/sum(1)"))

# COMMAND ----------

# MAGIC %md In the end this is still not an amazing accuracy but it is better than where be started and shows the power of pipelines on Spark

# COMMAND ----------


