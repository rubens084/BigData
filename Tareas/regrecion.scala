//importamos las clases y escojemos regrecion lojistica 
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
//permite esconder algunas alertas
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
//se crea la secion de spark
val spark = SparkS
//se imprime el esquema del data frame.se le indica el nombre y tipo de archivo este tiene que estar en 
//la carpeta dond e se encuentra el documento escaal o intrroducir la direccion.
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("titanic.csv")
data.printSchema()
//imprime la primera linia en el data freim indicando las diferentes casillas del esquema.
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example Data Row")
for(ind <- Range(1,colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
}
// se trata de asertar si algien sobrevivio o no en una catastrofe como lo fue el undimiento 
//del titanic tomando datos importantes que nos den como respuesta un si o un no.1 o 0
//se creara un datafreim seleccionando los datos de interes e ignorando los que no nos interesan.
val logregdataall = (data.select(data("Survived").as("label"),
$"Pclass",$"Name", $"Sex", $"Age", $"SibSp", $"Parch", $"Fare", $"Embarked"))
//hacemos drop a los datso que no se requerriran para este data set
val logregdata = logregdataall.na.drop()
//inportar estos para alludarnos con las columnas categoricas ONEHOTENCODER string = num
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

//combierte strings a datos numericos para ser mas facil su prosesamiento en este caso la columna
//sexo y la columna enbarque SEX y EMBARKED
val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
val embarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkIndex")                               

//comvertir datos numericos en 0 y 1s  One Hot Encoding para su facil manipulacion donde al programa
//sele es mas facil leer un numero 1 o 0  a un texto es mucho mas practico.
val genderEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
val embarkEncoder = new OneHotEncoder().setInputCol("EmbarkIndex").setOutputCol("EmbarkVec")

// (Label,features)
// Se crea un nuevo VectorAssembler, o indicamos como output y seleccionamos las columnas 
// a utilizar. 
val assembler = (new VectorAssembler()
                  .setInputCols(Array("Pclass","SexVec","Age","SibSp",
                   "Parch","Fare","EmbarkVec")).setOutputCol("features"))
//Dividimos nuestros en training y test set, Escojemos  los datos de logregdata que 
//los datos que requerrimos. ledesimos que el 70% 0.7 estara en el training y el 30% 
//del dataset estara en el test 0.3 
val Array(training,test) = logregdata.randomSplit(Array(0.7,0.3),seed=12345) 
//se utilisa Pipeline para ingresar los datos crudos por desirlo asi de una manera
import org.apache.spark.ml.Pipeline
val lr = new LogisticRegression()
//se crea un pipeline nuevo que nos ayudara con los indexers que creamos, se los agregaremos
//de la manera que los creamos 
val pipeline = new Pipeline().setStages(Array(genderIndexer,embarkIndexer,genderEncoder,embarkEncoder,assembler,lr))
// se lo alimentamos a los datos de training donde despues lo podremos comparar con los resultados del test
val model = pipeline.fit(training)

val results = model.transform(test)



/////  Evaluacion de modelo ///////////
// importamos multiclas metrics para la evalucacion del modulo
import org.apache.spark.mllib.evaluation.MulticlassMetrics
//a qui convertiremos solo requeriremos de la columna prediction  y lable  las creamos en double las dos y .rdd 
// podemos revisar el esquema con results.printSchema
val predictionAndLabels = results.select($"prediction",$"label").as[(Double,Double)].rdd

val metrics = new MulticlassMetrics(predictionAndLabels)
// al imprimir esto te muestar la matris de confucin 
println("Confusion matrix:")
println(metrics.confusionMatrix)
//al escrivir metrics. tabulador te mostrara las opciones que puedes utilizar
//las que requerimos son estas tres y dan el mismo resultado.
//metrics.accuracy,  metrics.recall  metrics.precision  

//para mostrar resultados ingresamos results.show() 
