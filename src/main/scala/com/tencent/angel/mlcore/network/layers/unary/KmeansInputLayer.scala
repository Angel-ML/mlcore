/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */
package com.tencent.angel.mlcore.network.layers.unary

import com.tencent.angel.ml.math2.matrix._
import com.tencent.angel.ml.math2.ufuncs.Ufuncs
import com.tencent.angel.ml.math2.utils.{DataBlock, LabeledData}
import com.tencent.angel.ml.math2.vector.{IntDoubleVector, IntFloatVector, _}
import com.tencent.angel.ml.math2.{MFactory, VFactory}
import com.tencent.angel.mlcore.conf.MLCoreConf
import com.tencent.angel.mlcore.network.layers.{InputLayer, Trainable}
import com.tencent.angel.mlcore.network.{Graph, TransFunc}
import com.tencent.angel.mlcore.optimizer.Optimizer
import com.tencent.angel.mlcore.utils.{LayerKeys, OptUtils}
import com.tencent.angel.mlcore.variable.{MatVariable, Variable, VecVariable}
import org.apache.commons.logging.LogFactory
import org.json4s.JsonAST.JField
import org.json4s.JsonDSL._

import scala.util.Random


class KmeansInputLayer(name: String,
                       outputDim: Int,
                       transFunc: TransFunc,
                       override val optimizer: Optimizer)(implicit graph: Graph)
  extends InputLayer(name, outputDim) with Trainable with Serializable {
  graph.addTrainableLayer(this)

  private val LOG = LogFactory.getLog(classOf[KmeansInputLayer])

  private val formatClassName = conf.getString(
    MLCoreConf.ML_SIMPLEINPUTLAYER_MATRIX_OUTPUT_FORMAT,
    MLCoreConf.DEFAULT_ML_SIMPLEINPUTLAYER_MATRIX_OUTPUT_FORMAT)
  private val center: MatVariable = graph.provider.getMatVariable(s"${name}_center", outputDim,
    conf.indexRange, optimizer, formatClassName, allowPullWithIndex = true)
  private val v: VecVariable = provider.getVecVariable(s"${name}_v", outputDim,
    null, formatClassName, allowPullWithIndex = false)
  private val indexRange = conf.indexRange
  private val C = conf.getDouble(MLCoreConf.KMEANS_C, MLCoreConf.DEFAULT_KMEANS_C)
  var vCount: Vector = _
  var vCopy: Vector = _

  override def toString: String = {
    s"KmeansInputLayer name=$name outputDim=$outputDim optimizer=$optimizer"
  }

  /**
    * Pick up K samples as initial centers randomly, and push them to PS.
    *
    * @param dataStorage : trainning data storage, the cluster center candidates
    */
  def initKCentersRandomly(totalTask: Int, dataStorage: DataBlock[LabeledData], K: Int): Unit = {
    LOG.info(s"Task[0] Initialize cluster centers with randomly choosen " +
      "samples.")
    val start = System.currentTimeMillis()
    val rand = new Random(System.currentTimeMillis())
    val initCenter = center.copy()

    for (i <- 0 until K) {
      if (i % totalTask == 0) {
        val newCent = dataStorage.get(rand.nextInt(dataStorage.size)).getX
        initCenter match {
          case mat: BlasDoubleMatrix =>
            mat.setRow(i, newCent)
          case mat: BlasFloatMatrix =>
            mat.setRow(i, newCent)
          case mat: RBIntDoubleMatrix =>
            mat.setRow(i, newCent.asInstanceOf[IntDoubleVector])
          case mat: RBIntFloatMatrix =>
            mat.setRow(i, newCent.asInstanceOf[IntFloatVector])
        }
      }
    }
    variableManager.putSlot(center.asInstanceOf[Variable], initCenter)
    variableManager.updateALL(0, 1)

    LOG.info(s"All tasks Init cluster centers success, cost ${System.currentTimeMillis() - start}" +
      s" ms")
  }

  override protected def doForward(input: Matrix): Matrix = {
    vCount = v.mul(1.0)
    vCopy = v.mul(1.0)
    val centerDist = center.mul(center).sum(1)
    val rowNum = input.getNumRows

    input match {
      case mat if mat.isInstanceOf[BlasDoubleMatrix] || mat.isInstanceOf[RBIntDoubleMatrix] =>
        val centerId = VFactory.denseDoubleVector(rowNum)
        val modelout = MFactory.denseDoubleMatrix(rowNum, indexRange.toInt + 1,
          new Array[Double](rowNum * (indexRange.toInt + 1)))
        val delta = MFactory.denseDoubleMatrix(rowNum, indexRange.toInt)
        for (i <- 0 until rowNum) {
          val x = mat.getRow(i)
          val len = x.dot(x)
          val dists = centerDist.sub(center.mul(x).mul(2).sum(1)).add(len)
          val id = dists.asInstanceOf[IntDoubleVector].argmin()

          centerId.set(i, id)
          delta.setRow(i, x.sub(center.getRow(id)))
        }
        modelout.setCol(0, centerId)
        for (i <- 1 until indexRange.toInt + 1) {
          modelout.setCol(i, delta.getCol(i - 1))
        }
        transFunc(modelout)

      case mat if mat.isInstanceOf[BlasFloatMatrix] || mat.isInstanceOf[RBIntFloatMatrix] =>
        val centerId = VFactory.denseFloatVector(rowNum)
        val modelout = MFactory.denseFloatMatrix(rowNum, indexRange.toInt + 1,
          new Array[Float](rowNum * (indexRange.toInt + 1)))
        val delta = MFactory.denseFloatMatrix(rowNum, indexRange.toInt)
        for (i <- 0 until rowNum) {
          val x = mat.getRow(i)
          val len = x.dot(x)
          val dists = centerDist.sub(center.mul(x).mul(2).sum(1)).add(len)
          val id = dists.asInstanceOf[IntFloatVector].argmin()

          centerId.set(i, id)
          delta.setRow(i, x.sub(center.getRow(id)))
        }
        modelout.setCol(0, centerId)
        for (i <- 1 until indexRange.toInt + 1) {
          modelout.setCol(i, delta.getCol(i - 1))
        }
        transFunc(modelout)
    }
  }


  override protected def doBackward(input: Matrix, gradInput: Matrix): Unit = {
    val transBack = transFunc.calGrad(forward(), gradInput)
    val rowNum = input.getNumRows
    val colNum = input.getRow(0).dim().toInt
    input match {
      case mat if mat.isInstanceOf[BlasDoubleMatrix] =>
        val deltaDist = MFactory.denseDoubleMatrix(outputDim, colNum)
        val delta = MFactory.denseDoubleMatrix(rowNum, colNum)
        val centerId = transBack.getCol(0).asInstanceOf[IntDoubleVector]
        for (i <- 0 until colNum) {
          delta.setCol(i, transBack.getCol(i + 1))
        }

        for (i <- 0 until delta.getNumRows) {
          vCount = vCount match {
            case vv: IntDoubleVector =>
              vv.set(centerId.get(i).toInt, vv.get(centerId.get(i).toInt) + 1)
              vv
            case vv: IntFloatVector =>
              vv.set(centerId.get(i).toInt, vv.get(centerId.get(i).toInt) + 1)
              vv
          }
          if (deltaDist.getRow(centerId.get(i).toInt).sum() == 0) {
            deltaDist.setRow(centerId.get(i).toInt, delta.getRow(i))
          } else {
            deltaDist.getRow(centerId.get(i).toInt).iadd(delta.getRow(i))
          }
        }
        delta.clear()

        val deltaV = OptUtils.wrapVector2Matrix(vCopy.sub(vCount))
        val deltaCenter = Ufuncs.divnonzero(deltaDist.mul(C), vCount.add(C), true)
        variableManager.putSlot(center.asInstanceOf[Variable], deltaCenter)
        variableManager.putSlot(v.asInstanceOf[Variable], deltaV)

      case mat if mat.isInstanceOf[BlasFloatMatrix] =>
        var deltaDist = MFactory.denseFloatMatrix(outputDim, colNum)
        val delta = MFactory.denseFloatMatrix(rowNum, colNum)
        val centerId = transBack.getCol(0).asInstanceOf[IntFloatVector]
        val centerCount = VFactory.denseIntVector(outputDim)
        for (i <- 0 until colNum) {
          delta.setCol(i, transBack.getCol(i + 1))
        }

        for (i <- 0 until delta.getNumRows) {
          centerCount.set(centerId.get(i).toInt, centerCount.get(centerId.get(i).toInt) + 1)
          vCount = vCount match {
            case vv: IntDoubleVector =>
              vv.set(centerId.get(i).toInt, vv.get(centerId.get(i).toInt) + 1)
              vv
            case vv: IntFloatVector =>
              vv.set(centerId.get(i).toInt, vv.get(centerId.get(i).toInt) + 1)
              vv
          }
          if (deltaDist.getRow(centerId.get(i).toInt).sum() == 0) {
            deltaDist.setRow(centerId.get(i).toInt, delta.getRow(i))
          } else {
            deltaDist.getRow(centerId.get(i).toInt).iadd(delta.getRow(i))
          }
        }
        delta.clear()

        val deltaV = OptUtils.wrapVector2Matrix(vCopy.sub(vCount))
        val deltaCenter = Ufuncs.divnonzero(deltaDist.mul(C), vCount.add(C), true)
        variableManager.putSlot(center.asInstanceOf[Variable], deltaCenter)
        variableManager.putSlot(v.asInstanceOf[Variable], deltaV)

      case mat if mat.isInstanceOf[RBIntDoubleMatrix] =>
        val deltaDist = MFactory.rbIntDoubleMatrix(outputDim, colNum)
        for (i <- 0 until outputDim) {
          deltaDist.setRow(i, VFactory.denseDoubleVector(colNum))
        }
        val delta = MFactory.denseDoubleMatrix(rowNum, colNum)
        val centerId = transBack.getCol(0).asInstanceOf[IntDoubleVector]
        for (i <- 0 until colNum) {
          delta.setCol(i, transBack.getCol(i + 1))
        }

        for (i <- 0 until delta.getNumRows) {
          vCount = vCount match {
            case vv: IntDoubleVector =>
              vv.set(centerId.get(i).toInt, vv.get(centerId.get(i).toInt) + 1)
              vv
            case vv: IntFloatVector =>
              vv.set(centerId.get(i).toInt, vv.get(centerId.get(i).toInt) + 1)
              vv
          }
          if (deltaDist.getRow(centerId.get(i).toInt).sum() == 0) {
            deltaDist.setRow(centerId.get(i).toInt, delta.getRow(i).asInstanceOf[IntDoubleVector])
          } else {
            deltaDist.getRow(centerId.get(i).toInt).iadd(delta.getRow(i))
          }
        }

        delta.clear()
        for (i <- 0 until outputDim) {
          deltaDist.getRow(i).imul(C).idiv(C + vCount.asInstanceOf[IntDoubleVector].get(i))
        }

        val deltaV = OptUtils.wrapVector2Matrix(vCopy.sub(vCount))
        variableManager.putSlot(center.asInstanceOf[Variable], deltaDist)
        variableManager.putSlot(v.asInstanceOf[Variable], deltaV)

      case mat if mat.isInstanceOf[RBIntFloatMatrix] =>
        val deltaDist = MFactory.rbIntFloatMatrix(outputDim, colNum)
        for (i <- 0 until outputDim) {
          deltaDist.setRow(i, VFactory.denseFloatVector(colNum))
        }
        val delta = MFactory.denseFloatMatrix(rowNum, colNum)
        val centerId = transBack.getCol(0).asInstanceOf[IntFloatVector]
        for (i <- 0 until colNum) {
          delta.setCol(i, transBack.getCol(i + 1))
        }

        for (i <- 0 until delta.getNumRows) {
          vCount = vCount match {
            case vv: IntDoubleVector =>
              vv.set(centerId.get(i).toInt, vv.get(centerId.get(i).toInt) + 1)
              vv
            case vv: IntFloatVector =>
              vv.set(centerId.get(i).toInt, vv.get(centerId.get(i).toInt) + 1)
              vv
          }
          if (deltaDist.getRow(centerId.get(i).toInt).sum() == 0) {
            deltaDist.setRow(centerId.get(i).toInt, delta.getRow(i).asInstanceOf[IntFloatVector])
          } else {
            deltaDist.getRow(centerId.get(i).toInt).iadd(delta.getRow(i))
          }
        }

        delta.clear()
        for (i <- 0 until outputDim) {
          deltaDist.getRow(i).imul(C).idiv(C + vCount.asInstanceOf[IntFloatVector].get(i))
        }

        val deltaV = OptUtils.wrapVector2Matrix(vCopy.sub(vCount))
        variableManager.putSlot(center.asInstanceOf[Variable], deltaDist)
        variableManager.putSlot(v.asInstanceOf[Variable], deltaV)

    }
  }

  private[mlcore] override def toJson: JField = {
    val layerJson = (LayerKeys.typeKey -> s"${this.getClass.getSimpleName}") ~
      (LayerKeys.outputDimKey -> outputDim) ~
      (LayerKeys.transFuncKey -> transFunc.toJson) ~
      (LayerKeys.optimizerKey -> optimizer.toJson)

    JField(name, layerJson)
  }
}