package com.tencent.angel.mlcore.optimizer

trait OptimizerProvider {
  def optFromJson(jsonStr: String): Optimizer

  def setRegParams[T <: Optimizer](opt: T, jastStr: String): T
}