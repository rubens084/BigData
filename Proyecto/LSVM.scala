// A support vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional 
// space, which can be used for classification, regression, or other tasks. Intuitively, a good separation is
// achieved by the hyperplane that has the largest distance to the nearest training-data points of any class 
//(so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}

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

// We modify the column "y" which is the output variable this indicates if the client will sign a term deposit how
// it will be classified based on this it has to be converted to numeric stringindexer will create a new column with 
// the values ​​of "and" but in numericbeing "0.0" for "no" and "1.0" for "yes"

val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
val dataIndexed = labelIndexer.fit(features).transform(features)


val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// Fit the model
val lsvcModel = lsvc.fit(dataIndexed)

// Print the coefficients and intercept for linear svc
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
