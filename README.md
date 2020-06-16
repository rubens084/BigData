# Bigdata
<br><strong>Unidad 4</strong>
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

<li>
   <li>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad4">Evaluacion</a>
<li><a href="https://github.com/rubens084/Bigdata/tree/Unidad4/Proyecto">Proyecto</a>



<li>
<li>
<li>This document contains exercises and practices of the kind of massive data taught in the technology of 
<li>Tijuana taught by Dr. Cristian Romero.
<li>the practices are taught in Spark in scala documents with a staggered learning system.
<li>
   <li>

<li>
<li> LSVM 
<li>A support vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional 
<li>space, which can be used for classification, regression, or other tasks. Intuitively, a good separation is
<li>achieved by the hyperplane that has the largest distance to the nearest training-data points of any class 
<li>(so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.
<li>import org.apache.spark.ml.classification.LinearSVC
<li>import org.apache.spark.sql.SparkSession
<li>import org.apache.log4j._
<li>import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
<li>
<li>help with some errors that may occur
<li>Logger.getLogger("org").setLevel(Level.ERROR)
<li>
<li>public static class SparkSession.Builder extends Object implements Logging
<li>val spark = SparkSession.builder().getOrCreate()
<li>
<li>load the data set and create the array with the necessary data
<li>val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
<li>
<li>This section covers algorithms for working with features, roughly divided into these groups:
<li>Extraction: Extracting features from “raw” data
<li>Transformation: Scaling, converting, or modifying features
<li>Selection: Selecting a subset from a larger set of features
<li>Locality Sensitive Hashing (LSH): This class of algorithms combines aspects of feature transformation with other algorithms.
<li>val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
<li>val features = assembler.transform(df)
<li>
<li>We modify the column "y" which is the output variable this indicates if the client will sign a term deposit how
<li>it will be classified based on this it has to be converted to numeric stringindexer will create a new column with 
<li>the values ​​of "and" but in numericbeing "0.0" for "no" and "1.0" for "yes"
<li>
<li>val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
<li>val dataIndexed = labelIndexer.fit(features).transform(features)
<li>
<li>
<li>val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
<li>
<li>Fit the model
<li>val lsvcModel = lsvc.fit(dataIndexed)
<li>
<li>Print the coefficients and intercept for linear svc
<li>println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
<li>
<li>
<li>
<li> ADT 

<li>The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset,
<li>and then evaluate on the held-out test set. We use two feature transformers to prepare the data; these help index categories 
<li>for the label and categorical features, adding metadata to the DataFrame which the Decision Tree algorithm can recognize.
<li>import org.apache.spark.sql.SparkSession
<li>import org.apache.log4j._
<li>import org.apache.spark.ml.Pipeline
<li>import org.apache.spark.ml.classification.DecisionTreeClassifier
<li>import org.apache.spark.ml.classification.DecisionTreeClassificationModel
<li>import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
<li>import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
<li>    
<li>help with some errors that may occur
<li>Logger.getLogger("org").setLevel(Level.ERROR)
<li>    
<li>public static class SparkSession.Builder extends Object implements Logging
<li>val spark = SparkSession.builder().getOrCreate()
    
<li>load the data set and create the array with the necessary data
<li>val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
<li>This section covers algorithms for working with features, roughly divided into these groups:
<li>Extraction: Extracting features from “raw” data
<li>Transformation: Scaling, converting, or modifying features
<li>Selection: Selecting a subset from a larger set of features
<li>Locality Sensitive Hashing (LSH): This class of algorithms combines aspects of feature transformation with other algorithms.
<li>val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
<li>val features = assembler.transform(df)
<li>    
<li>We modify the column "y" which is the output variable
<li>this indicates if the client will sign a term deposit
<li>how it will be classified based on this it has to be converted to numeric
<li>stringindexer will create a new column with the values ​​of "and" but in numeric
<li>being "0.0" for "no" and "1.0" for "yes"
<li>val labelIndexer0 = new StringIndexer().setInputCol("y").setOutputCol("label")
<li>val dataIndexed = labelIndexer0.fit(features).transform(features)
<li>    
<li>StringIndexer encodes a string column of labels to a column of label indices. The indices are in [0, numLabels), and four ordering options 
<li>are supported: “frequencyDesc”: descending order by label frequency (most frequent label assigned 0), “frequencyAsc”: ascending order by <li>label
<li>frequency (least frequent label assigned 0), “alphabetDesc”: descending alphabetical order, and “alphabetAsc”: ascending alphabetical <li>order 
<li>(default = “frequencyDesc”).
<li>val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)
<li>    
<li>VectorIndexer helps index categorical features in datasets of Vectors. It can both automatically decide which features are
<li>categorical and convert original values to category indices.
<li>We create automatic indexedFeatures with 4 categories
<li>val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)
<li>    
<li>We divide the data into an array into parts of 70% and 30%
<li>val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))
<li>    
<li>Train a DecisionTree model.
<li>val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
<li>    
<li>A Transformer that maps a column of indices back to a new column of corresponding string values. The index-string mapping is either from
<li>the ML attributes of the input column, or from user-supplied labels (which take precedence over ML attributes).
<li>Convert indexed labels back to original labels.
<li>val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
<li>    
<li>In this section, we introduce the concept of ML Pipelines. ML Pipelines provide a uniform set of high-level APIs built
<li>on top of DataFrames that help users create and tune practical machine learning pipelines.
<li>Chain indexers and tree in a Pipeline.
<li>val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
<li>    
<li>Train model. This also runs the indexers.
<li>val model = pipeline.fit(trainingData)
<li>    
<li>Make predictions.
<li>val predictions = model.transform(testData)
<li>    
<li>Select example rows to display.
<li>predictions.select("predictedLabel", "label", "features").show(10)
<li>    
<li>Evaluator for multiclass classification, which expects two input columns: prediction and label.
<li>Decision tree model for classification. It supports both binary and multiclass labels, as well as both continuous and categorical features.
<li>Select (prediction, true label) and compute test error.
<li>val evaluator = new <li>MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
<li>val accuracy = evaluator.evaluate(predictions)
<li>println(s"Test Error = ${(1.0 - accuracy)}\n")
<li>val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
<li>println(s"Learned classification tree model:\n\n ${treeModel.toDebugString}")
<li>
<li>

<li>  LR 

<li>LogisticRegression is the estimator of the pipeline. Following is the way to build the same logistic
<li>regression model by using the pipeline.
<li>import org.apache.spark.sql.SparkSession
<li>import org.apache.log4j._
<li>import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
<li>import org.apache.spark.ml.classification.LogisticRegression
<li>import org.apache.spark.mllib.evaluation.MulticlassMetrics
<li>import org.apache.spark.ml.Pipeline
<li>    
<li>help with some errors that may occur
<li>Logger.getLogger("org").setLevel(Level.ERROR)
<li>    
<li>public static class SparkSession.Builder extends Object implements Logging
<li>val spark = SparkSession.builder().getOrCreate()
<li>    
<li>load the data set and create the array with the necessary data
<li>val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
<li>This section covers algorithms for working with features, roughly divided into these groups:
<li>Extraction: Extracting features from “raw” data
<li>Transformation: Scaling, converting, or modifying features
<li>Selection: Selecting a subset from a larger set of features
<li>Locality Sensitive Hashing (LSH): This class of algorithms combines aspects of feature transformation with other algorithms.
<li>val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
<li>    
<li>We modify the column "y" which is the output variable
<li>this indicates if the client will sign a term deposit
<li>how it will be classified based on this it has to be converted to numeric
<li>stringindexer will create a new column with the values ​​of "and" but in numeric
<li>being "0.0" for "no" and "1.0" for "yes"
<li>val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
<li>val dataIndexed = labelIndexer.fit(df).transform(df)
<li>    
<li>We divide the data into an array into parts of 70% and 30%
<li>val Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 12345)
<li>    
<li>We create the new Logistic Regression lr 
<li>val lr = new LogisticRegression()
<li>In this section, we introduce the concept of ML Pipelines. ML Pipelines provide a uniform set of high-level APIs built
<li>on top of DataFrames that help users create and tune practical machine learning pipelines.
<li>We create the a pipeline
<li>val pipeline = new Pipeline().setStages(Array(assembler,lr))
<li>    
<li>Model the data, A fitted model, i.e., a Transformer produced by an Estimator.
<li>val model = pipeline.fit(training)
<li>    
<li>Results
<li>val results = model.transform(test)
<li>    
<li>Predictions
<li>val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
<li>    
<li>Metrics public class MulticlassMetrics extends Object Evaluator for multiclass classification
<li>param: predictionAndLabels an RDD of (prediction, label) pairs.
<li>val metrics = new MulticlassMetrics(predictionAndLabels)
<li>    
<li>Confusion matrix Returns confusion matrix: predicted classes are in columns, they are ordered by class label ascending, as in "labels"
<li>println(metrics.confusionMatrix)
<li>println(metrics.accuracy)
<li>
<li>

<li>  MLP     
<li>
<li>
<li>Multilayer perceptron classifier (MLPC) is a classifier based on the feedforward artificial neural network. MLPC consists of 
<li>multiple layers of nodes. Each layer is fully connected to the next layer in the network. Nodes in the input layer represent the input <li>data. 
<li>import org.apache.spark.sql.SparkSession
<li>import org.apache.log4j._
<li>import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
<li>import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
<li>import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
<li>import org.apache.spark.ml.linalg.Vectors
 <li>   
<li>help with some errors that may occur
<li>Logger.getLogger("org").setLevel(Level.ERROR)
    <li>
<li>public static class SparkSession.Builder extends Object implements Logging
<li>val spark = SparkSession.builder().getOrCreate()
<li>    
<li>load the data set and create the array with the necessary data
<li>val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
<li>This section covers algorithms for working with features, roughly divided into these groups:
<li>Extraction: Extracting features from “raw” data
<li>Transformation: Scaling, converting, or modifying features
<li>Selection: Selecting a subset from a larger set of features
<li>Locality Sensitive Hashing (LSH): This class of algorithms combines aspects of feature transformation with other algorithms.
<li>val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
<li>val features = assembler.transform(df)
 <li>   
<li>We modify the column "y" which is the output variable
<li>this indicates if the client will sign a term deposit
<li>how it will be classified based on this it has to be converted to numeric
<li>stringindexer will create a new column with the values ​​of "and" but in numeric
<li>being "0.0" for "no" and "1.0" for "yes"
<li>    
<li>val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
<li>val dataIndexed = labelIndexer.fit(features).transform(features)
<li>    
<li>We divide the data into an array into parts of 70% and 30%
<li>val split = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 1234L)
<li>val train = split(0)
<li>val test = split(1)
<li>    
<li>We specify the layers for the neural network entry 5 for the data number of the features
<li>2 hidden layers of two neurons and output 2 since it is only yes or no depending on whether 
<li>the client subscribed to a term deposit
<li>    
<li>val layers = Array[Int](5, 2, 3, 2)
<li>    
<li>We create the trainer with its parameters
<li>val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
    <li>
<li>We train the model
<li>val model = trainer.fit(train)
<li>    
<li>We print the accuracy The model.transform() method applies the same transformation to any
<li>new data with the same schema, and arrive at a prediction of how to classify the data.
<li>val result = model.transform(test)
<li>    
<li>predictions and label (original)
<li>val predictionAndLabels = result.select("prediction", "label")
<li>
<li>Model precision estimation runs
<li>val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
<li>println(s"Accuracy test = ${evaluator.evaluate(predictionAndLabels)}")




