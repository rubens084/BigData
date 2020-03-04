//Preparing Data & DataFrame
//Before, we start let’s create the DataFrame from a sequence of the data to work with.
// This DataFrame contains columns “employee_name”, “department”, “state“, “salary”, “age” and “bonus” columns.

//We will use this Spark DataFrame to run groupBy() on “department” columns and calculate aggregates
// like minimum, maximum, average, total salary for each group using min(), max() and sum() aggregate 
//functions respectively. and finally, we will also see how to do group and aggregate on multiple columns.


  import spark.implicits._
  val simpleData = Seq(("James","Sales","NY",90000,34,10000),
    ("Michael","Sales","NY",86000,56,20000),
    ("Robert","Sales","CA",81000,30,23000),
    ("Maria","Finance","CA",90000,24,23000),
    ("Raman","Finance","CA",99000,40,24000),
    ("Scott","Finance","NY",83000,36,19000),
    ("Jen","Finance","NY",79000,53,15000),
    ("Jeff","Marketing","CA",80000,25,18000),
    ("Kumar","Marketing","NY",91000,50,21000)
  )
  val df = simpleData.toDF("employee_name","department","salary","state","age","bonus")
  df.show()



//groupBy and aggregate on DataFrame columns
//Let’s do the groupBy() on department column of DataFrame and then find the sum of salary for each 
//department using sum() aggregate function.


df.groupBy("department").sum("salary").show(false)

//Similarly, we can calculate the number of employee in each department using count()


df.groupBy("department").count()
//Calculate the minimum salary of each department using min()


df.groupBy("department").min("salary")
//Calculate the maximin salary of each department using max()


df.groupBy("department").max("salary")
//Calculate the average salary of each department using avg()


df.groupBy("department").avg( "salary")
//Calculate the mean salary of each department using mean()


df.groupBy("department").mean( "salary") 
//groupBy and aggregate on multiple DataFrame columns
//Similarly, we can also run groupBy and aggregate on two or more DataFrame columns, below example does group
// by on department,state and does sum() on salary and bonus columns.


  //GroupBy on multiple columns
  df.groupBy("department","state")
    .sum("salary","bonus")
    .show(false)




//similarly, we can run group by and aggregate on tow or more columns for other aggregate functions,
// please refer below source code for example.

//Running more aggregates at a time
//Using agg() aggregate function we can calculate many aggregations at a time on a single statement using 
//Spark SQL aggregate functions sum(), avg(), min(), max() mean() e.t.c. In order to use these, we should import 
//"import org.apache.spark.sql.functions._"


import org.apache.spark.sql.functions._
  df.groupBy("department")
    .agg(
      sum("salary").as("sum_salary"),
      avg("salary").as("avg_salary"),
      sum("bonus").as("sum_bonus"),
      max("bonus").as("max_bonus"))
    .show(false)
//This example does group on department column and calculates sum() and avg() of salary for each department
// and calculates sum() and max() of bonus for each department.



//Using filter on aggregate data
//Similar to SQL “HAVING” clause, On Spark DataFrame we can use either where() or filter() function to filter
// the rows of aggregated data.


  df.groupBy("department")
    .agg(
      sum("salary").as("sum_salary"),
      avg("salary").as("avg_salary"),
      sum("bonus").as("sum_bonus"),
      max("bonus").as("max_bonus"))
    .where(col("sum_bonus") >= 50000)
    .show(false)
//This removes the sum of a bonus that has less than 50000 and yields below output.




package com.sparkbyexamples.spark.dataframe

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object GroupbyExample extends App {

  val spark: SparkSession = SparkSession.builder()
    .master("local[1]")
    .appName("SparkByExamples.com")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  val simpleData = Seq(("James","Sales","NY",90000,34,10000),
    ("Michael","Sales","NY",86000,56,20000),
    ("Robert","Sales","CA",81000,30,23000),
    ("Maria","Finance","CA",90000,24,23000),
    ("Raman","Finance","CA",99000,40,24000),
    ("Scott","Finance","NY",83000,36,19000),
    ("Jen","Finance","NY",79000,53,15000),
    ("Jeff","Marketing","CA",80000,25,18000),
    ("Kumar","Marketing","NY",91000,50,21000)
  )
  val df = simpleData.toDF("employee_name","department","state","salary","age","bonus")
  df.show()

  //Group By on single column
  df.groupBy("department").count().show(false)
  df.groupBy("department").avg("salary").show(false)
  df.groupBy("department").sum("salary").show(false)
  df.groupBy("department").min("salary").show(false)
  df.groupBy("department").max("salary").show(false)
  df.groupBy("department").mean("salary").show(false)

  //GroupBy on multiple columns
  df.groupBy("department","state")
    .sum("salary","bonus")
    .show(false)
  df.groupBy("department","state")
    .avg("salary","bonus")
    .show(false)
  df.groupBy("department","state")
    .max("salary","bonus")
    .show(false)
  df.groupBy("department","state")
    .min("salary","bonus")
    .show(false)
  df.groupBy("department","state")
    .mean("salary","bonus")
    .show(false)

  //Running Filter
  df.groupBy("department","state")
    .sum("salary","bonus")
    .show(false)

  //using agg function
  df.groupBy("department")
    .agg(
      sum("salary").as("sum_salary"),
      avg("salary").as("avg_salary"),
      sum("bonus").as("sum_bonus"),
      max("bonus").as("max_bonus"))
    .show(false)

  df.groupBy("department")
    .agg(
      sum("salary").as("sum_salary"),
      avg("salary").as("avg_salary"),
      sum("bonus").as("sum_bonus"),
      stddev("bonus").as("stddev_bonus"))
    .where(col("sum_bonus") > 50000)
    .show(false)
}