# Databricks notebook source
# MAGIC %md Read in the data
# MAGIC 
# MAGIC This data is located in in csv files and Apache Spark 2.0 can read the data in directly.

# COMMAND ----------

taxes2013 = spark.read\
  .option("header", "true")\
  .csv("dbfs:/databricks-datasets/data.gov/irs_zip_code_data/data-001/2013_soi_zipcode_agi.csv")

# COMMAND ----------

markets = spark.read\
  .option("header", "true")\
  .csv("dbfs:/databricks-datasets/data.gov/farmers_markets_geographic_data/data-001/market_data.csv")

# COMMAND ----------

display(taxes2013)

# COMMAND ----------

taxes2013.createOrReplaceTempView("taxes2013")
markets.createOrReplaceTempView("markets")

# COMMAND ----------

# MAGIC %sql show tables

# COMMAND ----------

# MAGIC %sql SELECT * FROM taxes2013 limit 5

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS cleaned_taxes;
# MAGIC 
# MAGIC CREATE TABLE cleaned_taxes AS
# MAGIC SELECT 
# MAGIC   state, 
# MAGIC   int(zipcode / 10) as zipcode,
# MAGIC   int(mars1) as single_returns,
# MAGIC   int(mars2) as joint_returns,
# MAGIC   int(numdep) as numdep,
# MAGIC   double(A02650) as total_income_amount,
# MAGIC   double(A00300) as taxable_interest_amount,
# MAGIC   double(a01000) as net_capital_gains,
# MAGIC   double(a00900) as biz_net_income
# MAGIC FROM taxes2013

# COMMAND ----------

# MAGIC %sql SELECT state, total_income_amount FROM cleaned_taxes 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC New Jersey and California have higher average incomes per zip code.

# COMMAND ----------

# MAGIC %sql describe cleaned_taxes

# COMMAND ----------

# MAGIC %md  Let's look at the set of zip codes with the lowest total capital gains and plot the results. 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT zipcode, SUM(net_capital_gains) AS cap_gains
# MAGIC FROM cleaned_taxes
# MAGIC   WHERE NOT (zipcode = 0000 OR zipcode = 9999)
# MAGIC GROUP BY zipcode
# MAGIC ORDER BY cap_gains ASC
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md Let's look at a combination of capital gains and business net income to see what we find. 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT zipcode,
# MAGIC   SUM(biz_net_income) as business_net_income,
# MAGIC   SUM(net_capital_gains) as capital_gains,
# MAGIC   SUM(net_capital_gains) + SUM(biz_net_income) as capital_and_business_income
# MAGIC FROM cleaned_taxes
# MAGIC   WHERE NOT (zipcode = 0000 OR zipcode = 9999)
# MAGIC GROUP BY zipcode
# MAGIC ORDER BY capital_and_business_income DESC
# MAGIC LIMIT 50

# COMMAND ----------

# MAGIC %sql
# MAGIC EXPLAIN
# MAGIC   SELECT zipcode,
# MAGIC     SUM(biz_net_income) as net_income,
# MAGIC     SUM(net_capital_gains) as cap_gains,
# MAGIC     SUM(net_capital_gains) + SUM(biz_net_income) as combo
# MAGIC   FROM cleaned_taxes
# MAGIC   WHERE NOT (zipcode = 0000 OR zipcode = 9999)
# MAGIC   GROUP BY zipcode
# MAGIC   ORDER BY combo desc
# MAGIC   limit 50

# COMMAND ----------

# MAGIC %sql CACHE TABLE cleaned_taxes

# COMMAND ----------

# MAGIC %md Running query again on cached table

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT zipcode,
# MAGIC   SUM(biz_net_income) as net_income,
# MAGIC   SUM(net_capital_gains) as cap_gains,
# MAGIC   SUM(net_capital_gains) + SUM(biz_net_income) as combo
# MAGIC FROM cleaned_taxes
# MAGIC   WHERE NOT (zipcode = 0000 OR zipcode = 9999)
# MAGIC GROUP BY zipcode
# MAGIC ORDER BY combo desc
# MAGIC limit 50

# COMMAND ----------

# MAGIC %md Now let's look at the Farmer's Market Data. 
# MAGIC 
# MAGIC Start with a total summation of farmer's markets per state. 

# COMMAND ----------

# MAGIC %sql SELECT State, COUNT(State) as Sum
# MAGIC       FROM markets 
# MAGIC       GROUP BY State 

# COMMAND ----------

# MAGIC %sql SELECT state, sum(total_income_amount) FROM cleaned_taxes group by 1

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- looking at what sates have the highest % of markets selling coffee
# MAGIC with aggs as (
# MAGIC   SELECT state
# MAGIC   , sum(case when Coffee = 'Y' then 1 else 0 end) as farmers_markets_with_coffee
# MAGIC   , count(*) as all_markets
# MAGIC   from markets 
# MAGIC   group by 1
# MAGIC   )
# MAGIC 
# MAGIC select state
# MAGIC , farmers_markets_with_coffee
# MAGIC , all_markets
# MAGIC , round(((farmers_markets_with_coffee * 1.0) / all_markets) * 100.0,1) as pct_that_sell_coffee
# MAGIC from aggs
# MAGIC order by 4 desc

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- looking at what sates have the highest % of markets selling coffee
# MAGIC with aggs as (
# MAGIC   SELECT state
# MAGIC   , sum(case when Coffee = 'Y' then 1 else 0 end) as farmers_markets_with_coffee
# MAGIC   , count(*) as all_markets
# MAGIC   from markets 
# MAGIC   group by 1
# MAGIC   )
# MAGIC 
# MAGIC select state
# MAGIC , farmers_markets_with_coffee
# MAGIC , all_markets
# MAGIC , round(((farmers_markets_with_coffee * 1.0) / all_markets) * 100.0,1) as pct_that_sell_coffee
# MAGIC from aggs
# MAGIC order by 4 desc

# COMMAND ----------


