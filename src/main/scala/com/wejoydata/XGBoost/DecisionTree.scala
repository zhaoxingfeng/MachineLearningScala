package com.wejoydata.XGBoost

import breeze.linalg._
import com.wejoydata.PythonLib.Numpy

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Map


/**
  * Created by zhaoxingfeng on 2019/3/13.
  */
object DecisionTree {

  /**
    * 定义一棵树，包含分裂节点信息、左右子树信息、叶子节点取值
    */
  class Tree() {
    var splitFeature: Int = 0
    var splitValue: Double = 0.0
    var splitGain: Double = 0.0
    var internalValue: Double = 0.0
    var nodeIndex: Int = 0
    var leafValue: Double = Double.PositiveInfinity
    var treeLeft: Tree = _
    var treeRight: Tree = _

    /**
      * 给定一条样本，计算在该棵上的预测值
      * @param dataSetOne 单条样本
      * @return
      */
    def predictLeafValue(dataSetOne: Array[Double]): Double = {
      if (this.leafValue != Double.PositiveInfinity) {
        this.leafValue
      } else if (dataSetOne(this.splitFeature) <= this.splitValue) {
        this.treeLeft.predictLeafValue(dataSetOne)
      } else {
        this.treeRight.predictLeafValue(dataSetOne)
      }
    }

    /**
      * 树结构转成json形式，便于直观查看
      * @return
      */
    def describeTree(): String = {
      if (this.treeLeft == null || this.treeRight == null) {
        "{leafValue:" + this.leafValue.toString + "}"
      } else {
        val leftInfo: String = this.treeLeft.describeTree()
        val rightInfo: String = this.treeRight.describeTree()
        val treeStructure: String =
          "{splitFeature:" + this.splitFeature.toString +
          ",splitValue:" + this.splitValue.toString +
          ",splitGain:" + this.splitGain.toString +
          ",internalValue:" + this.internalValue.toString +
          ",nodeIndex:" + this.nodeIndex.toString +
          ",leftTree:" + leftInfo +
          ",rightTree:" + rightInfo + "}"
        treeStructure
      }
    }
  }

  /**
    * 定义一棵决策树
    */
  class BasicDecisionTree(maxDepth: Int = 10, numLeaves: Int = 30, minSampleSplit: Int = 2, minSampleLeaf: Int = 1,
                          subSample: Double = 1.0, colSampleByTree: Double = 1.0, maxBin: Int = 225, minChildWeight: Double = 1.0,
                          regGamma: Double = 0.3, regLambda: Double = 0.3) {
    var decisionTree: Tree = _
    var nodeIndex = 0
    var decisionTreefeatureImportance: Map[Int, Int] = Map.empty

    def fit(dataSet: Array[Array[Double]], targets: Array[Array[Double]]): Tree = {
      decisionTree = _fit(dataSet, targets, 0)
      decisionTree
    }

    /**
      * 递归构造决策树
      * @param dataSet 训练样本
      * @param targets 训练样本对应的残差
      * @param depth 树深度
      * @return
      */
    def _fit(dataSet: Array[Array[Double]], targets: Array[Array[Double]], depth: Int): Tree = {
      if (dataSet.length <= minSampleSplit || targets.map(x => x(2)).sum <= minChildWeight) {
        val tree = new Tree()
        tree.leafValue = calcLeafValue(targets)
        tree
      } else if (depth < maxDepth) {
        var bestSplitFeature: Int = 0
        var bestSplitValue: Double = 0.0
        var bestSplitGain: Double = 0.0
        var bestInternalValue: Double = 0.0
        val bestSplit = chooseBestFeature(dataSet, targets)
        bestSplitFeature = bestSplit._1
        bestSplitValue = bestSplit._2
        bestSplitGain = bestSplit._3
        bestInternalValue = bestSplit._4

        val dataSetSplit = splitDataSet(dataSet, targets, bestSplitFeature, bestSplitValue)
        val leftDataSet = dataSetSplit._1
        val rightDataSet = dataSetSplit._2
        val leftTargets = dataSetSplit._3
        val rightTargets = dataSetSplit._4

        val tree = new Tree()
        if (leftDataSet.length <= minSampleLeaf || rightDataSet.length <= minSampleLeaf) {
          tree.leafValue = calcLeafValue(targets)
          tree
        } else {
          decisionTreefeatureImportance(bestSplitFeature) = decisionTreefeatureImportance.getOrElse(bestSplitFeature, 0) + 1
          tree.splitFeature = bestSplitFeature
          tree.splitValue = bestSplitValue
          tree.splitGain = bestSplitGain
          tree.internalValue = bestInternalValue
          tree.nodeIndex = nodeIndex
          nodeIndex += 1
          tree.treeLeft = _fit(leftDataSet, leftTargets, depth + 1)
          tree.treeRight = _fit(rightDataSet, rightTargets, depth + 1)
          tree
        }
      } else {
        val tree = new Tree()
        tree.leafValue = calcLeafValue(targets)
        tree
      }
    }

    /**
      * 寻找给定样本下最优的分裂特征和分裂取值
      * @param dataSet 给定样本
      * @param targets 给定需要拟合的目标值（一阶导、二阶导）
      * @return
      */
    def chooseBestFeature(dataSet: Array[Array[Double]], targets: Array[Array[Double]]): (Int, Double, Double, Double) = {
      var bestSplitGain = Double.NegativeInfinity
      var bestSplitFeature = 0
      var bestSplitValue = 0.0

      for (featureIndex <- dataSet(0).indices) {
        var uniqueValues: ArrayBuffer[Double] = ArrayBuffer.empty
        if (dataSet.map(x => x(featureIndex)).distinct.length <= 100) {
          uniqueValues ++= dataSet.map(x => x(featureIndex)).distinct
        } else {
          val dataSetColumn = dataSet.map(x => x(featureIndex)).sorted
          for (p <- linspace(0, 1, maxBin)) {
            uniqueValues += Numpy.percentile(dataSetColumn, p)
          }
        }

        for (splitValue <- uniqueValues) {
          val leftTargets: ArrayBuffer[Array[Double]] = ArrayBuffer.empty
          val rightTargets: ArrayBuffer[Array[Double]] = ArrayBuffer.empty
          for (index <- dataSet.indices) {
            if (dataSet(index)(featureIndex) <= splitValue) {
              leftTargets += targets(index)
            } else {
              rightTargets += targets(index)
            }
          }

          val splitGain = calcSplitGain(leftTargets.toArray, rightTargets.toArray)
          if (splitGain > bestSplitGain) {
            bestSplitFeature = featureIndex
            bestSplitValue = splitValue
            bestSplitGain = splitGain
          }
        }
      }
      val bestInternalValue = calcLeafValue(targets)
      (bestSplitFeature, bestSplitValue, bestSplitGain, bestInternalValue)
    }

    /**
      * 计算叶子节点取值
      * @param targets 目标值（一阶导、二阶导）
      * @return
      */
    def calcLeafValue(targets: Array[Array[Double]]): Double = {
      var sumGrad = 0.0
      var sumHess = 0.0
      targets.foreach(x => {
        sumGrad += x(1)
        sumHess += x(2)
      })
      - sumGrad / (sumHess + regLambda)
    }

    /**
      * 计算分裂增益
      * @param leftTargets 左树（一阶导、二阶导）
      * @param rightTargets 右树（一阶导、二阶导）
      * @return
      */
    def calcSplitGain(leftTargets: Array[Array[Double]], rightTargets: Array[Array[Double]]): Double = {
      val leftGrad = leftTargets.map(x => x(1)).sum
      val leftHess = leftTargets.map(x => x(2)).sum
      val rightGrad = rightTargets.map(x => x(1)).sum
      val rightHess = rightTargets.map(x => x(2)).sum

      val splitGain = 0.5 * (math.pow(leftGrad, 2) / (leftHess + regLambda) +
                            math.pow(rightGrad, 2) / (rightHess + regLambda) -
                          math.pow(leftGrad + rightGrad, 2) / (leftHess + rightHess + regLambda)) - regGamma
      splitGain
    }

    /**
      * 根据最优分裂特征和分裂点，划分现有数据集
      * @param dataSet 数据集
      * @param targets 目标值（一阶导、二阶导）
      * @param splitFeature 最优分裂特征
      * @param splitValue 最优分裂点
      * @return
      */
    def splitDataSet(dataSet: Array[Array[Double]], targets: Array[Array[Double]], splitFeature: Int, splitValue: Double) = {
      val leftDataSet = dataSet.filter {_(splitFeature) <= splitValue }
      val rightDataSet = dataSet.filter {_(splitFeature) > splitValue }

      var leftTargets: ArrayBuffer[Array[Double]] = ArrayBuffer.empty
      var rightTargets: ArrayBuffer[Array[Double]] = ArrayBuffer.empty
      for (index <- dataSet.indices) {
        if (dataSet(index)(splitFeature) <= splitValue) {
          leftTargets += targets(index)
        } else {
          rightTargets += targets(index)
        }
      }
      (leftDataSet, rightDataSet, leftTargets.toArray, rightTargets.toArray)
    }

    /**
      * 给定一条样本，求最终落到哪个叶子节点上
      * @param dataSetOne 一条样本
      * @return
      */
    def predict(dataSetOne: Array[Double]): Double = {
      decisionTree.predictLeafValue(dataSetOne)
    }
  }
}

