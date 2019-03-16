package com.wejoydata

import com.wejoydata.XGBoost.XGBoost.XGBClassifier

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object TestXGBoost {
  def main(args: Array[String]): Unit = {
    val filePath = "src/test/resources/pima indians.csv"
    val sourceFile = Source.fromFile(filePath, "utf-8")

    var dataSet: ArrayBuffer[Array[Double]] = ArrayBuffer.empty
    var labels: ArrayBuffer[Int] = ArrayBuffer.empty

    sourceFile.getLines.foreach(line => {
      val lineList = line.split(",")
      val dataSetOne = new Array[Double](lineList.length - 1)
      for (index <- 0 until lineList.length - 1) {
        dataSetOne(index) = lineList(index).toDouble
      }
      dataSet += dataSetOne
      labels += lineList(lineList.length - 1).toInt
    })

    val xgboostModel = new XGBClassifier(nEstimators = 50, maxDepth = 6, numLeaves = 40, learningRate=0.4,
      minSamplesSplit = 50, minSamplesLeaf = 20, subSample=1.0, colSampleByTree=1.0,
      maxBin=100, minChildWeight=1, regGamma=0.3, regLambda=0.4)
    xgboostModel.fit(dataSet.toArray, labels.toArray)

    val writer = new java.io.PrintWriter(new java.io.File("src/test/resources/result.csv"))
    for(p <- xgboostModel.predictProba(dataSet.toArray)) {
      writer.println(p(0) + "," + p(1))
    }
    writer.close()
  }

}
