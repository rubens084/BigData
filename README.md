# Bigdata
Bigdata
<br><strong>Unidad 3</strong>
<p align="center">
<br><strong>Tecnológico Nacional de México</strong>
<br><strong>Instituto Tecnológico de Tijuana</strong>
<br><strong>Subdirección académica</strong>
<br><strong>Departamento de Sistemas y Computación</strong>
<br><strong>Semestre: ENERO - JUNIO 2020</strong>
<br><strong>Ingeniería en Tecnologías de la Información y Comunicaciones</strong>
<br><strong>Ingeniería Informatica</strong>
<br><strong>Materia: Datos Masivos</strong>
<br><strong>Unidad: 3</strong>
<br><strong>Dorado Aguilus Ruben #15210328</strong>
   <br><strong>Mejia Manriquez Rocio #14212336</strong>
<br><strong>Docente: Dr. Jose Christian Romero Hernandez</strong>
</p>

### Unidad3
 


<li>
   <li>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad1/Tareas ">Unidad: 1</a>
<li>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad1/Tareas ">TAREAS</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Tareas/Practica1.scala  ">Practica 1</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Tareas/Practica2.scala ">Practica 2</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Tareas/Pearson%20correlation.txt ">Pearson</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Tareas/fibonacci.scala ">Fibonacci</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Tareas/TareaGroupBy.scala">Grup By</a>
<li>
   <li>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad1/Examen">EXAMEN</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Examen/Examen.scala">Examen 1</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Examen/ExamenU1-2.scala">Examen 2</a>
<li> master
<li>
   <li>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad3/Examen">EXAMEN</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad3/Examen/ExamenU3.scala">Examen-Wholesale-customers</a>



<li>
   <li>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad2 ">Unidad: 2</a>
<li>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad2/Tareas ">TAREAS</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Tareas/regrecion.scala  ">Regrecion linial</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Tareas/PIPELINE ">Pipeline</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Tareas/CONFUSION%20MATRIX ">Confusion matrix</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Tareas/Algorithms%20in%20Machine%20Learning ">Algoritmos M.L</a>

<li>
   <li>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad2">EXAMEN</a>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad2/Examen-iris">Examen-iris</a>

<li>
   <li>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad2/Exposiciones">Expociciones</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Exposiciones/LSVM.scala">Linear support vector machine</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Exposiciones/LR.scala">Logistic Regression</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Exposiciones/DT.scala">Decision Tree Classifier</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad2/Exposiciones/MLP.scala">Multilayer perceptron classifier</a>

<li>
   <li>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad3 ">Unidad: 3</a>
<li>
   <li>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad3/Examen">EXAMEN</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad3/Examen/ExamenU3.scala">Wholesale customers</a>

   <li>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad4 ">Unidad: 4</a>
<li>
   <li>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad4/Proyecto">Proyecto</a>
<li><a href="https://github.com/rubens084/Bigdata/blob/Unidad4/Proyecto/Full.scala">Comparacion</a>



<li>
<li>
<li>This document contains exercises and practices of the kind of massive data taught in the technology of 
<li>Tijuana taught by Dr. Cristian Romero.
<li>the practices are taught in Spark in scala documents with a staggered learning system.
<li>
   <li>


<li>implement scala.Serializable, java.io.Closeable, Logging
<li>The entry point to Spark programming with the Dataset and DataFrame API.
<li>In environments this has been created in advance (eg REPL, notebooks)
<li>import org.apache.spark.sql.SparkSession
<li>
<li>allows to hide some alerts
<li>import org.apache.log4j._
<li>Logger.getLogger("org").setLevel(Level.ERROR)
<li>
<li>is useful when applications want to share a SparkContext.
<li>So yes, you can use it to share a SparkContext object between applications.
<li>Yes, you can reuse broadcast variables and temporary tables in all parts.
<li>val spark = SparkSession.builder().getOrCreate()
<li>
<li>k-means is one of the most widely used grouping algorithms that groups data points
<li>in a predefined number of clusters. The MLlib implementation includes a parallel variant
<li>of the k-means ++ method called kmeans ||.
<li>KMeans is implemented as an Estimator and generates a KMeansModel as the base model.
<li>import org.apache.spark.ml.clustering.KMeans
<li>
<li>The data from the Wholesale Customers data.csv dataser is loaded in the variable "data"
<li>val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Wholesale customers data.csv")
<li>
<li>the columns are selected: Fresh, Milk, Groceries, Frozen, Detergents_Paper, Delicassen. and we proceed to create a set called feature_data
<li>val feature_data = data.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
<li>
<li>This section covers algorithms for working with features, roughly divided into these groups:
<li>Extraction: extraction of "raw" data characteristics
<li>Transformation: scale, convert or modify features
<li>Selection: select a subset of a larger feature set
<li>Locality Sensitive Hashing (LSH): This class of algorithms combines aspects of feature transformation with other algorithms.
<li>import org.apache.spark.ml.feature.VectorAssembler
<li>Factory methods for working with vectors. Note that dense vectors simply
<li>are rendered as NumPy array objects, so there is no need to convert them to use them in
<li>MLlib. For sparse vectors, the factory methods in this class create a type compatible with MLlib,
<li>or users can pass the column vectors scipy.sparse from SciPy.
<li>import org.apache.spark.ml.linalg.Vectors
<li>
<li>a new Vector Assembler object is created for the columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen. as an input set.
<li>val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", <li>"Delicassen")).setOutputCol("features")
<li>
<li>VectorAssembler is a transformer that combines a given list of columns into a single vector column.
<li>It is useful to combine raw features and features generated by different feature transformers in one
<li>vector of individual characteristics, to train ML models such as logistic regression and decision trees. VectorAssembler
<li>accepts the following types of input columns: all numeric types, boolean type and vector type. In every row
<li>the values ​​of the input columns will be concatenated into a vector in the specified order.
<li>val training_data = assembler.transform(feature_data).select($"features")
<li>
<li>k-means is one of the most widely used grouping algorithms that groups data points into
<li>a predefined number of clusters. The MLlib implementation includes a parallel variant of the k-means ++ method called kmeans ||.
<li>KMeans is implemented as an Estimator and generates a KMeansModel as the base model.
<li>val kmeans = new KMeans().setK(3).setSeed(1L)
<li>val model = kmeans.fit(training_data)
<li>
<li>MLlib supports grouping of k-means, one of the most widely used grouping algorithms that groups data points into predefined values
<li>number of groups. The MLlib implementation includes a parallel variant of the k-means ++ method called kmeans ||.
<li>val WSSEw = model.computeCost(training_data)
<li>println(s"Within set sum of Squared Errors = $WSSEw")
<li>
<li>Show the result.
<li>println("Cluster Centers: ")
<li>model.clusterCenters.foreach(println)





