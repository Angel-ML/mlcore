package com.tencent.angel.mlcore

import org.apache.commons.logging.LogFactory
import com.tencent.angel.ml.math2.utils.LabeledData
import com.tencent.angel.ml.math2.utils.DataBlock
import com.tencent.angel.mlcore.network.Graph
import com.tencent.angel.mlcore.optimizer.decayer.StepSizeScheduler

import scala.collection.mutable

trait Learner {
  private val LOG = LogFactory.getLog(classOf[Learner])

//  val model: GraphModel
//  val graph: Graph
  protected val ssScheduler: StepSizeScheduler

  protected def barrier(): Unit

  protected val preHook: mutable.ListBuffer[Learner.HookFunc] = new mutable.ListBuffer[Learner.HookFunc]()
  protected val postHook: mutable.ListBuffer[Learner.HookFunc] = new mutable.ListBuffer[Learner.HookFunc]()

  def addPreHook(func: Learner.HookFunc): Unit = {
    preHook.append(func)
  }

  def addPostHook(func: Learner.HookFunc): Unit = {
    postHook.append(func)
  }

  protected def trainOneEpoch(epoch: Int, iter: Iterator[Array[LabeledData]], numBatch: Int): Double

  def train(trainData: DataBlock[LabeledData], validationData: DataBlock[LabeledData]): MLModel = {
    train(trainData, null, validationData)
  }

  def train(posTrainData: DataBlock[LabeledData],
            negTrainData: DataBlock[LabeledData],
            validationData: DataBlock[LabeledData]): MLModel

  protected def validate(epoch: Int, valiData: DataBlock[LabeledData]): Unit
}

object Learner {
  type HookFunc = Graph => Unit
}
