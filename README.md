# Bigdata

<p align="center">
<br><strong>Tecnológico Nacional de México</strong>
<br><strong>Instituto Tecnológico de Tijuana</strong>
<br><strong>Subdirección académica</strong>
<br><strong>Departamento de Sistemas y Computación</strong>
<br><strong>Semestre: ENERO - JUNIO 2020</strong>
<br><strong>Ingeniería en Tecnologías de la Información y Comunicaciones</strong>
<br><strong>Ingeniería Informatica</strong>
<br><strong>Materia: Datos Masivos</strong>


<br><strong>Dorado Aguilus Ruben #15210328</strong>
   <br><strong>Mejia Manriquez Rocio #14212336</strong>
<br><strong>Docente: Dr. Jose Christian Romero Hernandez</strong>
</p>

### Unidad 1
  
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad1/Tareas ">TAREAS</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Tareas/Practica1.scala  ">Practica 1</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Tareas/Practica2.scala ">Practica 2</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Tareas/Pearson%20correlation.txt ">Pearson</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Tareas/fibonacci.scala ">Fibonacci</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Tareas/TareaGroupBy.scala">Grup By</a>


<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad1/Examen">EXAMEN</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Examen/Examen.scala">Examen 1</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Examen/ExamenU1-2.scala">Examen 2</a>

<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad1/Secciones">SECCION</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Secciones/Seccion_1.scala">Seccion 1</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Secciones/Seccion_2.scala">Seccion 2</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Secciones/Seccion_3.scala">Seccion 3</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Secciones/Seccion_4.scala">Seccion 4</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Secciones/Seccion_5.scala">Seccion 5</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Secciones/Seccion_6.scala">Seccion 6</a>


Develop an algorithm in scala that calculates the radius of a circle.
 simple operations, where the radius of a 15cm circle is obtained, with the formula diameter / 2 * pi
print("Circulo de 15cm ")     
```
15/(2*3.1416)
```
Develop an algorithm in scala that tells me if a number is a prime number.
Given the variable bird = "tweet", use string interpolation to print "I am writing a tweet".
```
val bird = "Tweet"
val greet = s"Estoy escribiendo en $bird"
```
play with variables that contains a word where you add it in which phrase
Given the variable message = "Hola Luke soy tu padre!" use slilce to extract the sequence "Luke".
using the st slice you can cut part of a phrase or a word giving the parameters
```
val st = "Hola Luke yo soy tu padre!"
st slice  (5,8)
```
What is the difference of a value and a variable in scala ?.
 ```
 Variables in scala
 Values (val) are immutable, once assigned cannot be changed.
 Variables (var) can be reallocated.
```
Note, when reassigning you must use the same type of data!
Given the tuple (2,4,5,1,2,3,3.1416,23) return the number 3.1416.
in a number array using._ you can extract a particular number
```
val my_tup = (2,4,5,1,2,3,3.1416,23)
my_tup._7
```


<a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Tareas/Practica2.scala ">Practica 2</a>


   1. Create a list called "list" with the elements "red", "white", "black"
   a list is created with val and the name by entering the data in this list
   ```
   val lista = List("rojo","blanco","negro")
 ```
  2. Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"
  to add more data to the list, just add at the end of it the data you want to enter
 ```
 val lista = List("rojo","blanco","negro","verde","amarillo","azul","naranja","perla")
```
 3. Bring the "list" "green", "yellow", "blue" items
 we use slice again to take part in the list
  ```
  lista slice (3,6)
  ```
  4. Create a number array in the 1-1000 range in 5-in-5 steps
   creating an arrangement is defined as fix.range and where it starts and where it 
   ends and how much the jumps will be
   ```
   Array.range(1,1000, 5)
   ```
   5. What are the unique elements of the List list (1,3,3,4,6,7,3,7) use conversion to sets
 ```
 var list2=List(1,3,3,4,6,7,3,7)
list2.toSet
```
6. Create a mutable map called names containing the following
 "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
 ```
 val names = collection.mutable.Map(("Jose", 20), ("Luis", 24), ("Ana", 23),("Susana",27))
 ```
  6 a. Print all map keys
 ```
 names.keys
 ```   
   b. Add the following value to the map ("Miguel", 23)
   ```
   names += ("Miguel" -> 23)
   ```     

<a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Tareas/Pearson%20correlation.txt ">Pearson's Correlation</a>

In statistics, Pearson's correlation coefficient is a linear measure between
two quantitative random variables. Unlike covariance, Pearson's correlation
is independent of the scale of measurement of the variables.
A Pearson correlation is a number between -1 and 1 that indicates the extent
to which two variables are linearly related. The Pearson correlation is also 
known as the “product moment correlation coefficient” (PMCC) or simply “correlation”.
 
 
 <li><li>.(a) ----------------------------------------------.(b)
     <li<li>>-
       <li><li>-
          <li>-
              <li><li>-
                 <li><li> -
                      <li>-   
                     <li> <li>  -   
                         <li><li>    .(c)

It is used to group data, in this case we have 3 points a, b, c where group a is a type
color like red and b is another type like green to which color does c belong if we 
go in the distance at a smaller distance is the one that belongs
   

   

<a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Tareas/fibonacci.scala">The Fibonacci sequence</a>

The Fibonacci sequence is, by definition, the integer sequence in which 
every number after the first two is the sum of the two preceding numbers.
To simplify: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,
The number of times the function is called causes a stack overflow in most languages.

Algorithm Downward recursive version....
If the number enterd is less than 2, the number will be returnd
if is less than 2, the function will do a series of operations, and  returns the result  
```
def  function(num: Int): Int =  {  
if (num<2)  {  
return num  }  
else  {  
return function(num-1) + function(num-2)  }  
}   
function(10) 
```
Version with explicit formula 
If the number enterd is less than 2, the number will be returnd
if is not less than 2, the proses of the formula is divided into parts to finally create the result.
```
def  function1(num: Double): Double =  {   
if (num<2)  {
return num}  
else  {  
var w = ((1+(Math.sqrt(5)))/2)  
var x = Math.pow(w,num)  
var y = Math.pow((1-w),num)  
var z = ((x-(y)))/(Math.sqrt(5))  
return(z)  }  }  
function1(10)
```

 In this algorithm a function was added that after having performed the corresponding 
 operations the function will give us a result (return) this must be an integer value (Int)
a cycle (for) starts where k = 1, will start cycling until it becomes (num)
(num) represents the value that will be entered into the function
Depending on the cycle (for) the variables (x, y, z) will begin to change
your result until the end of the cycle (for) The result will be returned with (return)
```
def function2 (num: Int): Int = {
var x = 0
var y = 1
var z = 0
for (k <- 1 to num)
{
z = y + x
x = y
y = z
}
 return (x)
}
function2 (10)
```
 In this a function was added after having performed the corresponding operations the function
 will give us a result. this must be an integer value (Int) A cycle (for) starts where k = 1,
 will start cycling until it becomes (num) this one represents the value that will be enterd 
 into the function Depending on the cycle (for) the variables (x, y) will begin to change their values
 until the end of the cycle (for).
```
def function3 (num: Int): Int = {
var x = 0
var y = 1
for (k <- 1 to num)
{
y = y + x
x = y - x
}
return (x)
}
function3 (10)
```
 In this algorithm a function asks for an value (Int) then return an integer value with decimals
 an array is created that starts from 0 to (num + 1) if the variable (num) is less than 2, 
 that same variable is returned as a result, Otherwise the vector with space (0) will have a 
 value of (0) and the vector with space (1) will have a value of (1) Start cycling with a for
 the vector The result will be the variable (num).
```
def function4 (num: Int): Double =
{
val vector = Array.range (0, num + 1)
if (num <2)
{
return (num)
}
else
{
vector (0) = 0
vector (1) = 1
for (k <- 2 to num)
{vector (k) = vector (k-1) + vector (k-2)
}return vector (num)
}}function4 (10)
```
 In this algorithm a function was added that after having performed the corresponding 
 operations the function will give us a result (return) this must be an integer value with 
 decimal(Double). If the value entered is less than or equal to 0, then that value 
 will be returned Otherwise you will have to make a series of operations Start a cycle 
 (while) where the variables will begin to change value depending on the cycle iteration
 If variable (i) is odd, different operations will be done If variable (i) is even, different 
 operations will be done Variable (i) will begin to change value each time the cycle is entered
 until you exit the cycle, and the sum of (a + b) is returned
```
def function5 (n: Double): Double =
{if (n <= 0){
return (n)}else{
var i: Double = n - 1
var auxOne: Double = 0
var auxTwo: Double = 1
var a: Double = auxTwo
var b: Double = auxOne
var c: Double = auxOne
var d: Double = auxTwo
while (i> 0){
if (i% 2 == 1){
auxOne = (d * b) + (c * a)
auxTwo = ((d + (b * a)) + (c * b))
a = auxOne
b = auxTwo}
else
{
var pow1 = Math.pow (c, 2)
var pow2 = Math.pow (d, 2)
auxOne = pow1 + pow2
auxTwo = (d * ((2 * (c)) + d))
c = auxOne
d = auxTwo}i = (i / 2)}
return (a + b)}}function5 (9)
```


<h4>Group by</h4>
<a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Tareas/TareaGroupBy.scala">Group by</a>

Preparing Data & DataFrame
Before, we start let’s create the DataFrame from a sequence of the data to work with.
This DataFrame contains columns “employee_name”, “department”, “state“, “salary”, “age” and “bonus” columns.

We will use this Spark DataFrame to run groupBy() on “department” columns and calculate aggregates
like minimum, maximum, average, total salary for each group using min(), max() and sum() aggregate 
functions respectively. and finally, we will also see how to do group and aggregate on multiple columns.

```
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
```


groupBy and aggregate on DataFrame columns
Let’s do the groupBy() on department column of DataFrame and then find the sum of salary for each 
department using sum() aggregate function.

```
df.groupBy("department").sum("salary").show(false)
```
Similarly, we can calculate the number of employee in each department using count()

```
df.groupBy("department").count()
```
Calculate the minimum salary of each department using min()
```
df.groupBy("department").min("salary")
```
Calculate the maximin salary of each department using max()
```
df.groupBy("department").max("salary")
```
Calculate the average salary of each department using avg()
```
df.groupBy("department").avg( "salary")
```
Calculate the mean salary of each department using mean()
```
df.groupBy("department").mean( "salary") 
```
groupBy and aggregate on multiple DataFrame columns
Similarly, we can also run groupBy and aggregate on two or more DataFrame columns, below example does group
by on department,state and does sum() on salary and bonus columns.

```
  GroupBy on multiple columns
  df.groupBy("department","state")
    .sum("salary","bonus")
    .show(false)
```



similarly, we can run group by and aggregate on tow or more columns for other aggregate functions,
please refer below source code for example.

Running more aggregates at a time
Using agg() aggregate function we can calculate many aggregations at a time on a single statement using 
Spark SQL aggregate functions sum(), avg(), min(), max() mean() e.t.c. In order to use these, we should import 

```
"import org.apache.spark.sql.functions._"

import org.apache.spark.sql.functions._
  df.groupBy("department")
    .agg(
      sum("salary").as("sum_salary"),
      avg("salary").as("avg_salary"),
      sum("bonus").as("sum_bonus"),
      max("bonus").as("max_bonus"))
    .show(false)
 ```
This example does group on department column and calculates sum() and avg() of salary for each department
and calculates sum() and max() of bonus for each department.



Using filter on aggregate data
Similar to SQL “HAVING” clause, On Spark DataFrame we can use either where() or filter() function to filter
the rows of aggregated data.

```
  df.groupBy("department")
    .agg(
      sum("salary").as("sum_salary"),
      avg("salary").as("avg_salary"),
      sum("bonus").as("sum_bonus"),
      max("bonus").as("max_bonus"))
    .where(col("sum_bonus") >= 50000)
    .show(false)
```
This removes the sum of a bonus that has less than 50000 and yields below output.

```
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
```
  Group By on single column
  ```
  df.groupBy("department").count().show(false)
  df.groupBy("department").avg("salary").show(false)
  df.groupBy("department").sum("salary").show(false)
  df.groupBy("department").min("salary").show(false)
  df.groupBy("department").max("salary").show(false)
  df.groupBy("department").mean("salary").show(false)
```
  GroupBy on multiple columns
  
  ```
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
```
  Running Filter
  ```
  df.groupBy("department","state")
    .sum("salary","bonus")
    .show(false)

  <li>using agg function
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

```


 
### Unidad2
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad2/Tareas ">Homework</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Tareas/regrecion.scala  ">Linear Regrecion</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Tareas/PIPELINE ">Pipeline</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Tareas/CONFUSION%20MATRIX ">Confusion matrix</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Tareas/Algorithms%20in%20Machine%20Learning ">Algo M.L</a>



<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad2 ">Unidad: 2</a>

<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad2/Tareas ">TAREAS</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Tareas/regrecion.scala  ">Regrecion linial</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Tareas/PIPELINE ">Pipeline</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Tareas/CONFUSION%20MATRIX ">Confusion matrix</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Tareas/Algorithms%20in%20Machine%20Learning ">Algoritmos M.L</a>

<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad2">EXAMEN</a>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad2/Examen-iris">Examen-iris</a>

<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad2/Exposiciones">Expociciones</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Exposiciones/LSVM.scala">Linear support vector machine</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Exposiciones/LR.scala">Logistic Regression</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Exposiciones/DT.scala">Decision Tree Classifier</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Exposiciones/MLP.scala">Multilayer perceptron classifier</a>


This document contains exercises and practices of the kind of massive data taught in the technology of 
Tijuana taught by Dr. Cristian Romero.
the practices are taught in Spark in scala documents with a staggered learning system.

### Exam Unit2

We add the necessary libraries to work with the algorithm Multilayer Perceptron.
Multilayer Perceptron Classifier (MLPC) is a neural network based classifier
artificial direct feeding. MLPC consists of multiple layers of nodes. Each layer is
fully connected to the next layer in the network.
```
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
```
Public class MulticlassClassificationEvaluator extends the evaluator implements DefaultParamsWritable
Evaluator for multiclass classification, which expects two input columns: prediction and label.
```
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
```
From the data set Iris.cvs, elaborate the necessary data cleaning by means of a scrip in scala spark,
we import the necessary libraries for cleaning.
A feature transformer that combines multiple columns into one vector column.
This requires one pass over the entire dataset. In case we need to infer
column lengths from the data, we require an additional call to the 'first' method
dataset, see parameter 'handleInvalid'.
```
import org.apache.spark.ml.feature.VectorAssembler
```
converts a single column to an index column (similar to a factor column in R)
```
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
```
Factory methods for working with vectors. Note that dense vectors simply
they are rendered as NumPy array objects, so there is no need to convert them to use them in
MLlib. For sparse vectors, the factory methods in this class create a type compatible with MLlib,
or users can pass the scipy.sparse column vectors from SciPy.
```
import org.apache.spark.ml.linalg.Vectors
```
The data from the dataser iris.csv is loaded in the variable "data"
```
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")
```
Null fields are removed drop rows that have null only in column onlyColumnInOneColumnDataFrame.
```
val dataClean = data.na.drop()
```
shows the name of the columns
```
data.schema.names
```
We see the scheme to check that all the values ​​are correctly classified in the dataset
```
data.printSchema()
```
the first 5 values of the list are shown with their data in a table
```
data.show(5)
```
The DESCRIBE FUNCTION statement returns the basic metadata information of an existing function.
The metadata information includes the function name, implementation class, and usage details.
If the optional EXTENDED option is specified, the basic metadata information is returned along with the extended usage information.
```
data.describe().show
```
A vector is declared that transforms the data to the variable "features" This section covers algorithms for working with features, <li> roughly divided into these groups:
Extraction: extraction of "raw" data characteristics, Transformation: scale, convert or modify characteristics Selection: select a subset of a larger set of characteristics

```
val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))
```

Features are transformed using the dataframe
```
val features = vectorFeatures.transform(dataClean)
```
A "StringIndexer" is declared that transforms the data in "species" into numerical data
```
val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")
```
We adjust the indexed species with the vector features
```
val dataIndexed = speciesIndexer.fit(features).transform(features)
```
With the variable "splits" we make a cut randomly 60% is used
of the dataset in training and 40% in test, a random cut with a seed of 1234 is used
```
val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)
```
The variable "train" is declared which will have 60% of the data in the position
(0) containing 60%
```
val train = splits(0)
```
The variable "test" is declared which will have 40% of the data in the potion (1)
with the remaining 40% of the data set

```
val test = splits(1)
```
The configuration of the layers for the artificial neural network model is established
```
val layers = Array[Int](4, 2, 2, 3)
```
The Multilayer algorithm trainer is configured with their respective parameters
```
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
```
Model is trained with training data
```
val model = trainer.fit(train)
```
They are tested already trained the model
```
val result = model.transform(test)
```
Select the prediction and the label that will be stored in the variable
```
val predictionAndLabels = result.select("prediction", "label")
```
Some data is displayed 
```
predictionAndLabels.s```how()
```
Model precision estimation runs
```
val evaluator = new 
```

### Unidad3

This document contains exercises and practices of the kind of massive data taught in the technology of 
Tijuana taught by Dr. Cristian Romero.
the practices are taught in Spark in scala documents with a staggered learning system.


<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad3 ">Unidad: 3</a>

<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad3/Examen">Exam</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad3/Examen/ExamenU3.scala">Wholesale customers</a>

#Exam code

Implement scala.Serializable, java.io.Closeable, Logging
The entry point to Spark programming with the Dataset and DataFrame API.
In environments this has been created in advance (eg REPL, notebooks)
```
import org.apache.spark.sql.SparkSession
allows to hide some alerts
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

```
Is useful when applications want to share a SparkContext.
So yes, you can use it to share a SparkContext object between applications.
Yes, you can reuse broadcast variables and temporary tables in all parts.
```
val spark = SparkSession.builder().getOrCreate()
```

k-means is one of the most widely used grouping algorithms that groups data points
in a predefined number of clusters. The MLlib implementation includes a parallel variant
of the k-means ++ method called kmeans ||.
KMeans is implemented as an Estimator and generates a KMeansModel as the base model.
```
import org.apache.spark.ml.clustering.KMeans
```
The data from the Wholesale Customers data.csv dataser is loaded in the variable "data"
```
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Wholesale customers data.csv")
```
the columns are selected: Fresh, Milk, Groceries, Frozen, Detergents_Paper, Delicassen. and we proceed to create a set called feature_data
```
val feature_data = data.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
```
This section covers algorithms for working with features, roughly divided into these groups:
Extraction: extraction of "raw" data characteristics
Transformation: scale, convert or modify features
Selection: select a subset of a larger feature set
Locality Sensitive Hashing (LSH): This class of algorithms combines aspects of feature transformation with other algorithms.
```
import org.apache.spark.ml.feature.VectorAssembler
```
Factory methods for working with vectors. Note that dense vectors simply
are rendered as NumPy array objects, so there is no need to convert them to use them in
MLlib. For sparse vectors, the factory methods in this class create a type compatible with MLlib,
or users can pass the column vectors scipy.sparse from SciPy.
```
import org.apache.spark.ml.linalg.Vectors
```
a new Vector Assembler object is created for the columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen. as an input set.
```
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")
```
VectorAssembler is a transformer that combines a given list of columns into a single vector column.
It is useful to combine raw features and features generated by different feature transformers in one
vector of individual characteristics, to train ML models such as logistic regression and decision trees. VectorAssembler
accepts the following types of input columns: all numeric types, boolean type and vector type. In every row
the values of the input columns will be concatenated into a vector in the specified order.
```
val training_data = assembler.transform(feature_data).select($"features")
```
k-means is one of the most widely used grouping algorithms that groups data points into
a predefined number of clusters. The MLlib implementation includes a parallel variant of the k-means ++ method called kmeans ||.
KMeans is implemented as an Estimator and generates a KMeansModel as the base model.
```
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(training_data)
```
MLlib supports grouping of k-means, one of the most widely used grouping algorithms that groups data points into predefined values
number of groups. The MLlib implementation includes a parallel variant of the k-means ++ method called kmeans ||.
```
val WSSEw = model.computeCost(training_data)
println(s"Within set sum of Squared Errors = $WSSEw")
```
###Show the result.
```
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
```







### Unidad4

<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad4 ">Unit: 4</a>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad4/Proyecto">Proyect</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad4/Proyecto/Full.scala">Comparation</a>

This document contains exercises and practices of the kind of massive data taught in the technology of 
Tijuana taught by Dr. Cristian Romero.
the practices are taught in Spark in scala documents with a staggered learning system.


### LSVM 
A support vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional 
space, which can be used for classification, regression, or other tasks. Intuitively, a good separation is
achieved by the hyperplane that has the largest distance to the nearest training-data points of any class 
(so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.
```
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
```
help with some errors that may occur
```
Logger.getLogger("org").setLevel(Level.ERROR)
```
public static class SparkSession.Builder extends Object implements Logging
```
val spark = SparkSession.builder().getOrCreate()
```
load the data set and create the array with the necessary data
```
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
```
This section covers algorithms for working with features, roughly divided into these groups:
Extraction: Extracting features from “raw” data
Transformation: Scaling, converting, or modifying features
Selection: Selecting a subset from a larger set of features
Locality Sensitive Hashing (LSH): This class of algorithms combines aspects of feature transformation with other algorithms.
```
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val features = assembler.transform(df)
```
We modify the column "y" which is the output variable this indicates if the client will sign a term deposit how
it will be classified based on this it has to be converted to numeric stringindexer will create a new column with 
the values of "and" but in numericbeing "0.0" for "no" and "1.0" for "yes"
```
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
val dataIndexed = labelIndexer.fit(features).transform(features)
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
<li>Fit the model
<li>val lsvcModel = lsvc.fit(dataIndexed)
```
Print the coefficients and intercept for linear svc
```
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
```

### ADT 

The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset,
and then evaluate on the held-out test set. We use two feature transformers to prepare the data; these help index categories 
for the label and categorical features, adding metadata to the DataFrame which the Decision Tree algorithm can recognize.
```
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
```    
help with some errors that may occur
```
Logger.getLogger("org").setLevel(Level.ERROR)
```    
public static class SparkSession.Builder extends Object implements Logging
```
val spark = SparkSession.builder().getOrCreate()
```    
load the data set and create the array with the necessary data
```
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
```
This section covers algorithms for working with features, roughly divided into these groups:
Extraction: Extracting features from “raw” data
Transformation: Scaling, converting, or modifying features
Selection: Selecting a subset from a larger set of features
Locality Sensitive Hashing (LSH): This class of algorithms combines aspects of feature transformation with other algorithms.
```
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val features = assembler.transform(df)
```   
We modify the column "y" which is the output variable
this indicates if the client will sign a term deposit
how it will be classified based on this it has to be converted to numeric
stringindexer will create a new column with the values ​​of "and" but in numeric
being "0.0" for "no" and "1.0" for "yes"
```
val labelIndexer0 = new StringIndexer().setInputCol("y").setOutputCol("label")
val dataIndexed = labelIndexer0.fit(features).transform(features)
```    
StringIndexer encodes a string column of labels to a column of label indices. The indices are in [0, numLabels), and four ordering options are supported: “frequencyDesc”: descending order by label frequency (most frequent label assigned 0), “frequencyAsc”: ascending order by label
frequency (least frequent label assigned 0), “alphabetDesc”: descending alphabetical order, and “alphabetAsc”: ascending alphabetical order 

```
(default = “frequencyDesc”).
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)
    
VectorIndexer helps index categorical features in datasets of Vectors. It can both automatically decide which features are
 ```
 
categorical and convert original values to category indices.
We create automatic indexedFeatures with 4 categories

```
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)
```    
We divide the data into an array into parts of 70% and 30%
```
<li>val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))
```    
Train a DecisionTree model.
```
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
```    
A Transformer that maps a column of indices back to a new column of corresponding string values. The index-string mapping is either from
the ML attributes of the input column, or from user-supplied labels (which take precedence over ML attributes).
Convert indexed labels back to original labels.
```
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
```   
In this section, we introduce the concept of ML Pipelines. ML Pipelines provide a uniform set of high-level APIs built
on top of DataFrames that help users create and tune practical machine learning pipelines.
Chain indexers and tree in a Pipeline.
```
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
```    
Train model. This also runs the indexers.
```
val model = pipeline.fit(trainingData)
```    
Make predictions.
```
val predictions = model.transform(testData)
```    
Select example rows to display.
```
predictions.select("predictedLabel", "label", "features").show(10)
```    
Evaluator for multiclass classification, which expects two input columns: prediction and label.
Decision tree model for classification. It supports both binary and multiclass labels, as well as both continuous and categorical features.
```
Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}\n")
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n\n ${treeModel.toDebugString}")
```


### LR 

LogisticRegression is the estimator of the pipeline. Following is the way to build the same logistic
regression model by using the pipeline.
```
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.Pipeline
```    
help with some errors that may occur
```
Logger.getLogger("org").setLevel(Level.ERROR)
```    
public static class SparkSession.Builder extends Object implements Logging
```
val spark = SparkSession.builder().getOrCreate()
```    
load the data set and create the array with the necessary data

```
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
```
This section covers algorithms for working with features, roughly divided into these groups:
Extraction: Extracting features from “raw” data
Transformation: Scaling, converting, or modifying features
Selection: Selecting a subset from a larger set of features
Locality Sensitive Hashing (LSH): This class of algorithms combines aspects of feature transformation with other algorithms.

```
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
```    
We modify the column "y" which is the output variable
this indicates if the client will sign a term deposit
how it will be classified based on this it has to be converted to numeric
stringindexer will create a new column with the values ​​of "and" but in numeric
being "0.0" for "no" and "1.0" for "yes"

```
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
val dataIndexed = labelIndexer.fit(df).transform(df)
```    
We divide the data into an array into parts of 70% and 30%
```
val Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 12345)
```    
We create the new Logistic Regression lr 
```
val lr = new LogisticRegression()
```
In this section, we introduce the concept of ML Pipelines. ML Pipelines provide a uniform set of high-level APIs built
on top of DataFrames that help users create and tune practical machine learning pipelines.
We create the a pipeline
```
val pipeline = new Pipeline().setStages(Array(assembler,lr))
```   
Model the data, A fitted model, i.e., a Transformer produced by an Estimator.
```
val model = pipeline.fit(training)
```    
Results
```
val results = model.transform(test)
```    
Predictions
```
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
```    
Metrics public class MulticlassMetrics extends Object Evaluator for multiclass classification
param: predictionAndLabels an RDD of (prediction, label) pairs.
```
val metrics = new MulticlassMetrics(predictionAndLabels)
```    
Confusion matrix Returns confusion matrix: predicted classes are in columns, they are ordered by class label ascending, as in "labels"
```
println(metrics.confusionMatrix)
println(metrics.accuracy)
```


### MLP     

Multilayer perceptron classifier (MLPC) is a classifier based on the feedforward artificial neural network. MLPC consists of 
multiple layers of nodes. Each layer is fully connected to the next layer in the network. Nodes in the input layer represent the input data. 
```
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
 ```   
help with some errors that may occur
```
Logger.getLogger("org").setLevel(Level.ERROR)
```
public static class SparkSession.Builder extends Object implements Logging
```
val spark = SparkSession.builder().getOrCreate()
```    
load the data set and create the array with the necessary data
```
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
```
This section covers algorithms for working with features, roughly divided into these groups:
Extraction: Extracting features from “raw” data
Transformation: Scaling, converting, or modifying features
Selection: Selecting a subset from a larger set of features
Locality Sensitive Hashing (LSH): This class of algorithms combines aspects of feature transformation with other algorithms.
```
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val features = assembler.transform(df)
 ```   
We modify the column "y" which is the output variable
this indicates if the client will sign a term deposit
how it will be classified based on this it has to be converted to numeric
stringindexer will create a new column with the values of "and" but in numeric
being "0.0" for "no" and "1.0" for "yes"
```    
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
val dataIndexed = labelIndexer.fit(features).transform(features)
```    
We divide the data into an array into parts of 70% and 30%
```
val split = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 1234L)
<li>val train = split(0)
<li>val test = split(1)
```    
We specify the layers for the neural network entry 5 for the data number of the features
2 hidden layers of two neurons and output 2 since it is only yes or no depending on whether 
the client subscribed to a term deposit

```   
val layers = Array[Int](5, 2, 3, 2)
```    
We create the trainer with its parameters

```
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
 ```
We train the model

```
val model = trainer.fit(train)
```    
We print the accuracy The model.transform() method applies the same transformation to any
new data with the same schema, and arrive at a prediction of how to classify the data.

```
val result = model.transform(test)
```    
predictions and label (original)
```
val predictionAndLabels = result.select("prediction", "label")
```
Model precision estimation runs
```
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Accuracy test = ${evaluator.evaluate(predictionAndLabels)}")
```


This document contains exercises and practices of the kind of massive data taught in the technology of 
Tijuana taught by Dr. Cristian Romero.
the practices are taught in Spark in scala documents with a staggered learning system.


