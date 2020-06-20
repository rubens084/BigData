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
<br><strong>Unidad: 4</strong>
<br><strong>Dorado Aguilus Ruben #15210328</strong>
   <br><strong>Mejia Manriquez Rocio #14212336</strong>
<br><strong>Docente: Dr. Jose Christian Romero Hernandez</strong>
</p>

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





