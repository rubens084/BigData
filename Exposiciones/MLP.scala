    // Multilayer perceptron classifier (MLPC) is a classifier based on the feedforward artificial neural network. MLPC consists of 
    // multiple layers of nodes. Each layer is fully connected to the next layer in the network. Nodes in the input layer represent the input data. 
    import org.apache.spark.sql.SparkSession
    import org.apache.log4j._
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
    import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.linalg.Vectors
    
    // help with some errors that may occur
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    // public static class SparkSession.Builder extends Object implements Logging
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
    
    val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
    val dataIndexed = labelIndexer.fit(features).transform(features)
    
    // We divide the data into an array into parts of 70% and 30%
    val split = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 1234L)
    val train = split(0)
    val test = split(1)
    
    // We specify the layers for the neural network entry 5 for the data number of the features
    // 2 hidden layers of two neurons and output 2 since it is only yes or no depending on whether 
    // the client subscribed to a term deposit
    
    val layers = Array[Int](5, 2, 3, 2)
    
    // We create the trainer with its parameters
    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
    
    // We train the model
    val model = trainer.fit(train)
    
    // We print the accuracy The model.transform() method applies the same transformation to any
    // new data with the same schema, and arrive at a prediction of how to classify the data.
    val result = model.transform(test)
    
    // predictions and label (original)
    val predictionAndLabels = result.select("prediction", "label")

    // Model precision estimation runs
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    println(s"Accuracy test = ${evaluator.evaluate(predictionAndLabels)}")
    
  