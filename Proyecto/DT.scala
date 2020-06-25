    // The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset,
    // and then evaluate on the held-out test set. We use two feature transformers to prepare the data; these help index categories 
    // for the label and categorical features, adding metadata to the DataFrame which the Decision Tree algorithm can recognize.
    import org.apache.spark.sql.SparkSession
    import org.apache.log4j._
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.DecisionTreeClassifier
    import org.apache.spark.ml.classification.DecisionTreeClassificationModel
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
    
    // // help with some errors that may occur
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    // // public static class SparkSession.Builder extends Object implements Logging
    val spark = SparkSession.builder().getOrCreate()
    
    // load the data set and create the array with the necessary data
    val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
    // This section covers algorithms for working with features, roughly divided into these groups:
    // Extraction: Extracting features from “raw” data
    // Transformation: Scaling, converting, or modifying features
    // Selection: Selecting a subset from a larger set of features
    // Locality Sensitive Hashing (LSH): This class of algorithms combines aspects of feature transformation with other algorithms.
    val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
    val features = assembler.transform(df)
    
    // We modify the column "y" which is the output variable
    // this indicates if the client will sign a term deposit
    // how it will be classified based on this it has to be converted to numeric
    // stringindexer will create a new column with the values ​​of "and" but in numeric
    // being "0.0" for "no" and "1.0" for "yes"
    val labelIndexer0 = new StringIndexer().setInputCol("y").setOutputCol("label")
    val dataIndexed = labelIndexer0.fit(features).transform(features)
    
    // StringIndexer encodes a string column of labels to a column of label indices. The indices are in [0, numLabels), and four ordering options 
    // are supported: “frequencyDesc”: descending order by label frequency (most frequent label assigned 0), “frequencyAsc”: ascending order by label
    // frequency (least frequent label assigned 0), “alphabetDesc”: descending alphabetical order, and “alphabetAsc”: ascending alphabetical order 
    // (default = “frequencyDesc”).
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)
    
    // VectorIndexer helps index categorical features in datasets of Vectors. It can both automatically decide which features are
    // categorical and convert original values to category indices.
    // We create automatic indexedFeatures with 4 categories
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)
    
    // We divide the data into an array into parts of 70% and 30%
    val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))
    
    // Train a DecisionTree model.
    val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
    
    // A Transformer that maps a column of indices back to a new column of corresponding string values. The index-string mapping is either from
    // the ML attributes of the input column, or from user-supplied labels (which take precedence over ML attributes).
    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    
    // In this section, we introduce the concept of ML Pipelines. ML Pipelines provide a uniform set of high-level APIs built
    // on top of DataFrames that help users create and tune practical machine learning pipelines.
    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
    
    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)
    
    // Make predictions.
    val predictions = model.transform(testData)
    
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(10)
    
    // Evaluator for multiclass classification, which expects two input columns: prediction and label.
    // Decision tree model for classification. It supports both binary and multiclass labels, as well as both continuous and categorical features.
    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}\n")
    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n\n ${treeModel.toDebugString}")
    
    
    
  
    
    
  