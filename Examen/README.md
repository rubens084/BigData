
<a https://github.com/rubens084/Bigdata/blob/Unidad1/Examen/Examen.scala">Examn 1 </a>

<li>An array of 3-story numbers is generated which must be declared, subsequently
<li>define tuples to be able to continue with the exercise. it is necessary to take diagonally the
<li>Fix numbers to solve this problem you must know the number of boxes
<li>handle in the matrix.
<li>Diagonal difference is then declared to be the result of the subtraction of the
<li>diagonal and as a final result the name of the result is given.
<li>as a final result it gives 15.

val arre= ((11, 2, 4), (4, 5, 6), (10, 8, -12))
def Sumadiagonales (arreglotuplas: ((Int, Int, Int), (Int, Int, Int), (Int, Int, Int))) : Int ={
val diagonal_1 = (arre._1._1) + (arre._2._2) + (arre._3._3)
val diagonal_2 = (arre._1._3) + (arre._2._2) + (arre._3._1)
var diagonaldifference = diagonal_1 - diagonal_2
var resultado = math.abs (diagonaldifference)
return resultado
}
Sumadiagonales(arre)




<a https://github.com/rubens084/Bigdata/blob/Unidad1/Examen/ExamenU1-2.scala ">Examn 2 </a>
                               
<li>The entry point to programming Spark with the Dataset and DataFrame API                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession   
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
<li>2 Cargue el archivo
<li>The Spark driver program uses it to connect to the cluster manager to communicate, 
<li>submit Spark jobs and knows what resource manager (YARN, Mesos or Standalone) to communicate to.
val spark = SparkSession.builder().getOrCreate()
//2 Cargue el archivo
<li>it is used to load the dataset in this case and sun cvs file with netfliz content
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
<li>The datatype specified for id in the schema is Long but when schema is printed it is cast to String
df.printSchema()
//3 Mostrar nombre de columnas
<li>show the columns contained in the dataframe
df.columns
<li>4 Esquema
<li>show the complete data frame with the data
df.show()
<li>5 Mostrar las primeras 5 columnas
<li>select the columns you want to display and use the .show command
df.select("Date","Open","High","Low","Close").show()
<li>6 usa Describe() para aprender sobre el dataframe
<li>show relevant information about the column
df.describe ("High").show 
<li>7 Crea un nuevo dataframe con la columna HV ratio
<li>the new data frame is created with the new name and it is indicated what columns it will have and the name of these

val df2 = df.withColumn("HV Ratio", df("High")/ df("Volume"))

df2.show

  <li>8 Que dato es el pico mas alto en la columna close
<li>the command to select the maximum within the column is given and to show it we use the show column

import org.apache.spark.sql.SparkSession   
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")

df.groupBy("Date").max("Close").show()

<li>9 Cual es el significado de la columna close
<li>Descripcion:Es el valor de cierre de las acciones de netflix.
<li>10 cual es el maximo y minimo de la columna volumen
<li>similar as the previous one we select we indicate that it will be the maximum or the minimum 
<li>column which we want to find the value and the show command to show the data
df.select(max("Volume")).show()
df.select(min("Volume")).show() 

<li>11 Con sitaxis/spark contesta lo siguiente
<li>a) Cuantos dias fue la columna close inferior a $600
df.filter($"Close" < 600).count()
<li>b Que % del tiempo fue la columna high mayor a $500
df.filter($"High">500).count() * 1.0/df.count()*100
<li>c Cual es la correlacion de pearson entre columna high y volumen
df.select(corr($"High", $"Volume")).show()
<li>d Cual es el maximo de la columa high por a;o
df.select(max("High")).show()

<li>e Cual es el promedio de la columna close para cada  mes del calendario
df.select(avg("Close")).show() 
