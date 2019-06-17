import org.apache.spark.sql.SparkSession
import spark.implicits._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._ 
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.input_file_name
import org.apache.spark.sql.functions.udf

import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics 
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.Pipeline

/* Path of data files */
val path_Dataset = "../HAPT Data Set/RawData"

/*
* Setting Neural Network
*/
val numFeatures = 6
val numClasses = 12

val trainSet = 0.7
val testSet = 0.3

/* Create a SparkSession */
val spark = SparkSession.
            builder().
            appName("Project BDP").
            getOrCreate()

/*  
* Function for cleaning the name of files
*/
def cleanNameFile(x: String): String = { 
  try{
      val str = x.split("/")(9)
      return str.split("_")(1).
             concat("_").
             concat(str.split("_")(2)).
             dropRight(4).
             toString
  } catch {
      case e: Exception => ""
  }
}

val getNameClean = udf(cleanNameFile _)

val schemaAcceleration = StructType(Array(
                         StructField("acceleration_x", DoubleType, nullable = true), 
                         StructField("acceleration_y", DoubleType, nullable = true), 
                         StructField("acceleration_z", DoubleType, nullable = true),
                         StructField("file_name_A", StringType, nullable = true)
                      ))

val schemaGyroscope = StructType(Array(
                      StructField("gyroscope_x", DoubleType, nullable = true), 
                      StructField("gyroscope_y", DoubleType, nullable = true), 
                      StructField("gyroscope_z", DoubleType, nullable = true),
                      StructField("file_name_B", StringType, nullable = true)
                    ))

val schemaLabelsReady = StructType(Array(
                        StructField("Id_Experiment", IntegerType, nullable = true),
                        StructField("Index", IntegerType, nullable = true), 
                        StructField("label", DoubleType, nullable = true)
                      ))


println("-------------------------------------------------------")
println("----------------- READING DATA FILES ------------------")
println("-------------------------------------------------------")

val dfAcceleration_tmp = spark.read.
                         option("header", false).
                         option("inferSchema", false).
                         option("delimiter", " ").
                         schema(schemaAcceleration).
                         csv(path_Dataset + "/acc*").
                         select("acceleration_x","acceleration_y","acceleration_z").
                         withColumn("file_name_A", getNameClean(input_file_name()))
  
val dfAcceleration = dfAcceleration_tmp.
                     orderBy($"file_name_A").
                     withColumn("indexA",monotonically_increasing_id+1)

val dfGyroscope_tmp = spark.read.
                      option("header", false).
                      option("inferSchema", false).
                      option("delimiter", " ").
                      schema(schemaGyroscope).
                      csv(path_Dataset + "/gyro*").
                      select("gyroscope_x","gyroscope_y","gyroscope_z").
                      withColumn("file_name_B", getNameClean(input_file_name()))

val dfGyroscope = dfGyroscope_tmp.
                  orderBy($"file_name_B").
                  withColumn("indexB", monotonically_increasing_id+1)

val dfLabelsReady = spark.read.
                    option("header", false).
                    option("inferSchema", false).
                    option("delimiter", " ").
                    schema(schemaLabelsReady).
                    csv("../tmp/labels_ready.txt").
                    select("Id_Experiment", "Index","label")

/* 
* Join dfLabelsReady with dfGyroscope and dfAcceleration
*/
val dfNuevo = dfLabelsReady.
              join(dfAcceleration, 
              dfAcceleration("indexA") === dfLabelsReady("Index"),
              "inner")

val dfReady_tmp = dfNuevo.join(dfGyroscope, 
                  dfNuevo("indexA") === dfGyroscope("indexB"), "inner").
                  drop(dfNuevo("file_name_A")).
                  drop(dfGyroscope("file_name_B")).
                  drop(dfNuevo("indexA")).
                  drop(dfNuevo("Id_Experiment")).
                  drop(dfGyroscope("indexB"))

/*
*	Final dataset with the features and target value. I filtered label > 0
*/
val dfReady = dfReady_tmp.
              filter($"label" > 0 ).
              orderBy($"indexA").
              drop($"Index")

println("-------------------------------------------------------")
println("-------------- STARTING MACHINE LEARNING --------------")
println("-------------------------------------------------------")

/*
*	Using VectorAssembler(), To create a vector with all features.
*/
val assembler = new VectorAssembler().
	            setInputCols(Array("acceleration_x", "acceleration_y", 
                "acceleration_z", "gyroscope_x", "gyroscope_y", "gyroscope_z")).
	            setOutputCol("features")
val features = assembler.transform(dfReady)

/*
*	Using StandardScaler(), To Standard all the features using 
*	Standard Deviation.
*/
val scaler = new StandardScaler().
				setInputCol("features").
				setOutputCol("scaledFeatures").
  			setWithStd(true).
  			setWithMean(false)

val scalerModel = scaler.fit(features)
val scaledData = scalerModel.transform(features)

/*
*	I created a Index to label and features columns using 
*	StringIndexer() and VectorIndexer().
*/
val labelIndexer = new StringIndexer().
                   setInputCol("label").
                   setOutputCol("indexedLabel").
                   fit(scaledData)

val featureIndexer = new VectorIndexer().
                     setInputCol("scaledFeatures").
                     setOutputCol("indexedFeatures").
                     setMaxCategories(numFeatures).
                     fit(scaledData)

/*
*	To split the data between train and test. For that I used
*	two global variables.
*/
val splits = scaledData.
             randomSplit(Array(trainSet, testSet))

val trainingData = splits(0)
val testData = splits(1)

/*
*	This function is to create a model given an architecture of neural
*	network. Return the predictions to testData.
*/
def CreateModel(layers: Array[Int]): org.apache.spark.sql.Dataset[_] = {

  val trainer = new MultilayerPerceptronClassifier().
                setLayers(layers).
                setLabelCol("indexedLabel").
                setFeaturesCol("indexedFeatures").
                setBlockSize(128).
                setSeed(System.currentTimeMillis).
                setMaxIter(200)

  val labelConverter = new IndexToString().
                     setInputCol("prediction").
                     setOutputCol("predictedLabel").
                     setLabels(labelIndexer.labels)

  val pipeline = new Pipeline().
               setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))

  val model = pipeline.fit(trainingData)

  return model.transform(testData)
}

println("-------------------------------------------------------")
println("--------------- CREATING NEURAL NETWORK ---------------")
println("-------------------------------------------------------")

/* Neural network configuration */
val layers = Array[Int](numFeatures, 15, 20, 15, numClasses)

/* Called a function CreateModel and return test prediction */
val predictions = CreateModel(layers)

println("-------------------------------------------------------")
println("----------------------- RESULTS -----------------------")
println("-------------------------------------------------------")

/* Multilayer Perceptron Classifier Evaluation */
val evaluator = new MulticlassClassificationEvaluator().
                setLabelCol("indexedLabel").
                setPredictionCol("prediction").
                setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

/*
* --------------------------------------------
*				SOME RESULTS
* --------------------------------------------

val layers = Array[Int](6, 20, 20, 20, 12)
accuracy: Double = 0.6840659340659341 

val layers = Array[Int](6, 18, 20, 18, 12)
accuracy: Double = 0.6911586596689544

val layers = Array[Int](6, 17, 20, 17, 12) ***** BEST ******
accuracy: Double = 0.703482382031733
with noise = accuracy: Double = 0.5847007722007722 

val layers = Array[Int](6, 16, 20, 16, 12)
accuracy: Double = 0.7063686466625843 

val layers = Array[Int](6, 30, 40, 30, 12)
accuracy: Double = 0.6524953789279113 

val layers = Array[Int](6, 30, 40, 40, 30, 12)
accuracy: Double = 0.6638655462184874 

val layers = Array[Int](6, 15, 20, 15, 12)
accuracy: Double = 0.689877300613497

val layers = Array[Int](6, 8, 8, 12)
accuracy: Double = 0.6477379095163807    

val layers = Array[Int](6, 7, 12)
accuracy: Double = 0.62701062215478                                        

val layers = Array[Int](6, 8, 12)
accuracy: Double = 0.6122257053291537
*/




