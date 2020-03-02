import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
df.columns
df.show()
df.printSchema()
df.select("Date","Open","High","Low","Close").show()
df.describe().show()
val df2 = df.withColumn("HV Ratio", df("High")+ df("Volume")) 
df.select(max("Close")).show() 
println("Son los valores de cierre de las inversiones de Netflix durante los dias analizados")
df.select(min(df("Volume"))).show()
df.select(max(df("Volume"))).show()
df.filter($"Close"<600).count()
df.filter($"High">500).count() * 1.0 / df.count() * 100
df.select(corr($"High", $"Volume")).show()
val dfyear = df.withColumn("Year",year(df("Date")))
val maxyear = dfyear.select($"Year", $"High").groupBy("Year").max()
val res = maxyear.select($"Year", $"max(High)")
res.show()
val dfmonth = df.withColumn("Month",month(df("Date")))
val avgmonth = dfmonth.select($"Month",$"Close").groupBy("Month").mean()
avgmonth.select($"Month",$"avg(Close)").orderBy("Month").show()
