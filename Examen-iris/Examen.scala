// Agregamos las lbrerias necesarias para trabajar con el algortimo Multilayer Perceptron.
// El clasificador de perceptrón multicapa (MLPC) es un clasificador basado en la red neuronal
// artificial de alimentación directa. MLPC consta de múltiples capas de nodos. Cada capa está 
// completamente conectada a la siguiente capa en la red.
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
// Clase pública MulticlassClassificationEvaluatorextiende el evaluadorimplementa DefaultParamsWritable
// Evaluador para clasificación multiclase, que espera dos columnas de entrada: predicción y etiqueta.
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Del data set Iris.cvs, elaborar la limpieza de datos necesaria por medio de un scrip en scala spark, 
// impportamos las librerias necesarias para la limpieza.
// Un transformador de características que combina varias columnas en una columna vectorial.
// Esto requiere una pasada sobre todo el conjunto de datos. En caso de que necesitemos inferir 
// longitudes de columna a partir de los datos, requerimos una llamada adicional al 'primer' método 
// de conjunto de datos, consulte el parámetro 'handleInvalid'.
import org.apache.spark.ml.feature.VectorAssembler
// convierte una sola columna en una columna de índice (similar a una columna de factor en R)
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
// Métodos de fábrica para trabajar con vectores. Tenga en cuenta que los vectores densos simplemente 
// se representan como objetos de matriz NumPy, por lo que no es necesario convertirlos para usarlos en 
// MLlib. Para vectores dispersos, los métodos de fábrica en esta clase crean un tipo compatible con MLlib,
// o los usuarios pueden pasar los vectores de columna scipy.sparse de SciPy.
import org.apache.spark.ml.linalg.Vectors

// Se cargan los datos del dataser iris.csv en la variable "data"
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")
// Se eliminan los campos null suelta las filas que tienen nulo solo en la columna onlyColumnInOneColumnDataFrame.
val dataClean = data.na.drop()
//muestra el nombre de las columnas
data.schema.names
// Vemos el esquema para comprobar que todos los valores estan calsificados correctamente en el dataset
data.printSchema()
// se muestran los primeros 5 valores de la lista con sus datos en una tabla
data.show(5)
// La instrucción DESCRIBE FUNCTION devuelve la información básica de metadatos de una función existente. 
// La información de metadatos incluye el nombre de la función, la clase de implementación y los detalles de uso. 
// Si se especifica la opción EXTENDED opcional, la información básica de metadatos se devuelve junto con la información de uso extendida.
data.describe().show

// Se declara un vector que se transforma los datos a la variable "features" Esta sección cubre algoritmos para trabajar con características, divididas aproximadamente en estos grupos:
// Extracción: extracción de características de datos "en bruto", Transformación: escalar, convertir o modificar características
// Selección: seleccionar un subconjunto de un conjunto más amplio de características
val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))


// Se transforman los features usando el dataframe
val features = vectorFeatures.transform(dataClean)

// Se declara un "StringIndexer" que transformada los datos en "species" en datos numericos 
val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")

// Ajustamos las especies indexadas con el vector features
val dataIndexed = speciesIndexer.fit(features).transform(features)

// Con la variable "splits" hacemos un corte de forma aleatoria se utiliza el 60%
// del dataset en training y 40% en test, se utiliza un corte random con una semilla de 1234
val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)

// Se declara la variable "train" la cual tendra el 60% de los datos en la posicion
//(0) que contiene el 60%
val train = splits(0)

// Se declara la variable "test" la cual tendra el 40% de los datosen la pocicion (1)
// con el sobrante 40% del data set
val test = splits(1)

// Se establece la configuracion de las capas para el modelo de redes neuronales artificiales
val layers = Array[Int](4, 2, 2, 3)

// Se configura el entrenador del algoritmo Multilayer con sus respectivos parametros
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// Se entrena el modelo con los datos de entrenamiento
val model = trainer.fit(train)

// Se prueban ya entrenado el modelo
val result = model.transform(test)

// Se selecciona la prediccion y la etiqueta que seran guardado en la variable 
val predictionAndLabels = result.select("prediction", "label")

// Se muestran algunos datos 
predictionAndLabels.show()

// Se ejecuta la estimacion de la precision del modelo
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictionAndLabels)

// Se imprime el error del modelo
println(s"Test Error = ${(1.0 - accuracy)}")



