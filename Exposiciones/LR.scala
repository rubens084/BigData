    // LogisticRegression is the estimator of the pipeline. Following is the way to build the same logistic
    // regression model by using the pipeline.
    import org.apache.spark.sql.SparkSession
    import org.apache.log4j._
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
    import org.apache.spark.ml.classification.LogisticRegression
    import org.apache.spark.mllib.evaluation.MulticlassMetrics
    import org.apache.spark.ml.Pipeline
    
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
    
    // We modify the column "y" which is the output variable
    // this indicates if the client will sign a term deposit
    // how it will be classified based on this it has to be converted to numeric
    // stringindexer will create a new column with the values ​​of "and" but in numeric
    // being "0.0" for "no" and "1.0" for "yes"
    val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
    val dataIndexed = labelIndexer.fit(df).transform(df)
    
    // We divide the data into an array into parts of 70% and 30%
    val Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 12345)
    
    // We create the new Logistic Regression lr 
    val lr = new LogisticRegression()
    // In this section, we introduce the concept of ML Pipelines. ML Pipelines provide a uniform set of high-level APIs built
    // on top of DataFrames that help users create and tune practical machine learning pipelines.
    // We create the a pipeline
    val pipeline = new Pipeline().setStages(Array(assembler,lr))
    
    // Model the data, A fitted model, i.e., a Transformer produced by an Estimator.
    val model = pipeline.fit(training)
    
    // Results
    val results = model.transform(test)
    
    // Predictions
    val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
    
    // Metrics public class MulticlassMetrics extends Object Evaluator for multiclass classification
    //param: predictionAndLabels an RDD of (prediction, label) pairs.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    
    // Confusion matrix Returns confusion matrix: predicted classes are in columns, they are ordered by class label ascending, as in "labels"
    println(metrics.confusionMatrix)
    println(metrics.accuracy)
    