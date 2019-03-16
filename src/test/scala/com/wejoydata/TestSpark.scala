package com.wejoydata

import org.apache.spark.sql.SparkSession

object TestSpark {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local[2]").appName("XGBoost").getOrCreate()
    import spark.implicits._

    val dataPath = "src/test/resources/pima indians.csv"
    val savePath = "src/test/resources/pima indians1.csv"
    val df = spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", true.toString)
      .csv(dataPath)
    df.show(5)

    df.select(df("number"), df("age")).write.mode("overwrite").save(savePath)
    //    df.select(df("number"), df("age")).repartition(1).write.mode("overwrite").format("csv").option("header", true).save(savePath)
    df.select(df("number") < 10).show()
    val usersDF = spark.read.load(savePath)
    usersDF.select("number", "age").show()
    usersDF.show()

    df.select($"number", $"age").show(5)
  }

}
