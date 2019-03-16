package com.wejoydata.PythonLib

/**
  * Created by zhaoxingfeng on 2019/3/14.
  */
object Numpy {
  /**
    * 求分位数，仿照numpy.percentile
    * @param arrayInput
    * @param p
    * @return
    */
  def percentile(arrayInput: Array[Double], p: Double): Double = {
    if (p == 0) {
      arrayInput(0)
    } else if (100 - p < 0.00001) {
      arrayInput(arrayInput.length - 1)
    } else {
      val temp = (arrayInput.length - 1) * p / 100.0 + 1
      val posInteger = temp.toInt
      val posDecimal = temp - posInteger
      arrayInput(posInteger - 1) + (arrayInput(posInteger) - arrayInput(posInteger - 1)) * posDecimal
    }
  }
}
