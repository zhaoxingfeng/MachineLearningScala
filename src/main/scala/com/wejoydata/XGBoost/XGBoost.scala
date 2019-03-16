package com.wejoydata.XGBoost

import com.wejoydata.XGBoost.DecisionTree.BasicDecisionTree
import com.wejoydata.XGBoost.DecisionTree.Tree

import scala.collection.mutable.Map

/**
  * Created by zhaoxingfeng on 2019/3/13.
  */
object XGBoost {
  class XGBClassifier(nEstimators: Int = 100, maxDepth: Int = -1, numLeaves: Int = -1, learningRate: Double = 0.1,
                      minSamplesSplit: Int = 2, minSamplesLeaf: Int = 1, subSample: Double = 1.0, colSampleByTree: Double = 1.0,
                      maxBin: Int = 225, minChildWeight: Double = 1.0, regGamma: Double = 0.3, regLambda: Double = 0.3) {
    var xgboostTrees: Map[Int, Tree] = Map.empty
    var featureImportances: Map[Int, Int] = Map.empty
    var yPred_0 = 0.0

    /**
      * xgboost训练入口
      * @param dataSet 训练样本
      * @param labelInput 样本标签
      */
    def fit(dataSet: Array[Array[Double]], labelInput: Array[Int]): Unit = {
      if (labelInput.distinct.length != 2) {
        println("There must be two class for label!")
        throw new IllegalArgumentException
      }

      val meanValue = labelInput.sum.toDouble / labelInput.length
      yPred_0 = 0.5 * math.log((1 + meanValue) / (1 - meanValue))

      var targets = new Array[Array[Double]](labelInput.length)
      for (index <- labelInput.indices) {
        targets(index) = calcGradHess(labelInput(index), yPred_0)
      }

      for (stage <- 0 until nEstimators) {
        println("================================" + stage + "================================")

        val baseDecisionTree = new BasicDecisionTree(maxDepth, numLeaves, minSamplesSplit, minSamplesLeaf, subSample,
                                                    colSampleByTree, maxBin, minChildWeight, regGamma, regLambda)
        val treeStage = baseDecisionTree.fit(dataSet, targets)
        println(treeStage.describeTree())
        xgboostTrees += (stage -> treeStage)

        for (index <- labelInput.indices) {
          val yPred = targets(index)(0) + learningRate * treeStage.predictLeafValue(dataSet(index))
          val gradHess = calcGradHess(labelInput(index), yPred)
          targets(index) = Array(yPred, gradHess(1) + targets(index)(1), gradHess(2) + targets(index)(2))
        }

        for ((key, value) <- baseDecisionTree.decisionTreefeatureImportance) {
          featureImportances(key) = featureImportances.getOrElse(key, 0) + value
        }
      }
      println(featureImportances.toList.sortBy(_._1))
    }

    /**
      * 输入预测样本，计算p值
      *
      * @param dataSet 预测样本
      * @return
      */
    def predictProba(dataSet: Array[Array[Double]]): Array[Array[Double]] = {
      var outputResult = new Array[Array[Double]](dataSet.length)
      for (index <- dataSet.indices) {
        var fValue = yPred_0
        for (stage <- 0 until nEstimators) {
          fValue += learningRate * xgboostTrees(stage).predictLeafValue(dataSet(index))
        }
        val p_0 = Math.round(1.0 / (1 + math.exp(2 * fValue)) * 10000) / 10000.0
        val p_1 = Math.round((1 - p_0) * 100) / 100.0
        outputResult(index) = Array(p_0, p_1)
      }
      outputResult
    }

    /**
      * 计算一阶和二阶导数
      *
      * @param y 真实标签
      * @param yPred 预测值
      * @return
      */
    def calcGradHess(y: Double, yPred: Double): Array[Double] = {
      val pred = 1.0 / (1.0 + math.exp(-yPred))
      val grad = (-y + (1 - y) * math.exp(pred)) / (1 + math.exp(pred))
      val hess = math.exp(pred) / math.pow(1 + math.exp(pred), 2)
      Array(yPred, grad, hess)
    }
  }
}
