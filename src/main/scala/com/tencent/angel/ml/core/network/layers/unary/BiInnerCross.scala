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

package com.tencent.angel.ml.core.network.layers.unary

import com.tencent.angel.ml.core.network.Graph
import com.tencent.angel.ml.core.network.layers._
import com.tencent.angel.ml.core.utils.MLException
import com.tencent.angel.ml.math2.matrix._
import com.tencent.angel.ml.math2.vector._
import com.tencent.angel.ml.math2.{MFactory, VFactory}
import org.apache.commons.logging.LogFactory


class BiInnerCross(name: String, outputDim: Int, inputLayer: Layer)(implicit graph: Graph)
  extends LinearLayer(name, outputDim, inputLayer) {
  val LOG = LogFactory.getLog(classOf[BiInnerCross])

  override protected def doForward(input: Matrix): Matrix = {
    val batchSize = graph.getBatchSize
    input match {
      case mat: RBCompIntDoubleMatrix =>
        val data: Array[Double] = new Array[Double](batchSize * outputDim)
        (0 until batchSize).foreach { row =>
          val partitions = mat.getRow(row).getPartitions
          var opIdx = 0
          partitions.zipWithIndex.foreach { case (vector_outter, cidx_outter) =>
            if (cidx_outter != partitions.length - 1) {
              ((cidx_outter + 1) until partitions.length).foreach { cidx_inner =>
                data(row * outputDim + opIdx) = vector_outter.dot(partitions(cidx_inner))
                opIdx += 1
              }
            }
          }
        }
        MFactory.denseDoubleMatrix(batchSize, outputDim, data)
      case mat: RBCompIntFloatMatrix =>
        val data: Array[Float] = new Array[Float](batchSize * outputDim)
        (0 until batchSize).foreach { row =>
          val partitions = mat.getRow(row).getPartitions
          var opIdx = 0
          partitions.zipWithIndex.foreach { case (vector_outter, cidx_outter) =>
            if (cidx_outter != partitions.length - 1) {
              ((cidx_outter + 1) until partitions.length).foreach { cidx_inner =>
                data(row * outputDim + opIdx) = vector_outter.dot(partitions(cidx_inner)).toFloat
                opIdx += 1
              }
            }
          }
        }
        MFactory.denseFloatMatrix(batchSize, outputDim, data)
      case _ => throw MLException("ERROR! Only Comp Matrix is supported!")
    }
  }

  override protected def doBackward(input: Matrix, gradInput: Matrix): Matrix = {
    conf.valueType() match {
      case "double" =>
        val inputData = input.asInstanceOf[RBCompIntDoubleMatrix]

        val gradRows = inputData.getRows.zipWithIndex.map { case (compVector, row) =>
          val rowGrad = gradInput.getRow(row)
          val partGrad = (0 until compVector.getNumPartitions).toArray.map { i =>
            val grad = getValidateGrad(rowGrad.asInstanceOf[IntKeyVector], i)
            val mat = getMatrixFromCompVector(compVector, i)
            mat.transDot(grad).asInstanceOf[IntDoubleVector]
          }
          VFactory.compIntDoubleVector(compVector.getDim, partGrad)
        }

        MFactory.rbCompIntDoubleMatrix(gradRows)
      case "float" =>
        val inputData = input.asInstanceOf[RBCompIntFloatMatrix]

        val gradRows = inputData.getRows.zipWithIndex.map { case (compVector, row) =>
          val rowGrad = gradInput.getRow(row)
          val partGrad = (0 until compVector.getNumPartitions).toArray.map { i =>
            val grad = getValidateGrad(rowGrad.asInstanceOf[IntKeyVector], i)
            val mat = getMatrixFromCompVector(compVector, i)
            mat.transDot(grad).asInstanceOf[IntFloatVector]
          }
          VFactory.compIntFloatVector(compVector.getDim, partGrad)
        }

        MFactory.rbCompIntFloatMatrix(gradRows)
      case _ => throw MLException("Only Double and Float are support!")
    }
  }

  private def getValidateGrad(row: IntKeyVector, idx: Int): Vector = {
    val numFeild = Math.ceil(Math.sqrt(2.0 * row.getDim)).toInt
    val const = numFeild * 2 - 1

    val idxsPair = if (idx == 0) {
      (idx + 1 until numFeild).map(x2 => (idx, x2))
    } else if (idx == numFeild - 1) {
      (0 until idx).map(x1 => (x1, idx))
    } else {
      (0 until idx).map(x1 => (x1, idx)) ++ (idx + 1 until numFeild).map(x2 => (idx, x2))
    }

    val idxs = idxsPair.toArray.map { case (x1: Int, x2: Int) =>
      ((const - x1) * x1 / 2.0 + (x2 - x1) - 1).toInt
    }

    row match {
      case v: IntDoubleVector =>
        VFactory.denseDoubleVector(idxs.map(i => v.get(i)))
      case v: IntFloatVector =>
        VFactory.denseFloatVector(idxs.map(i => v.get(i)))
    }
  }

  private def getMatrixFromCompVector(row: ComponentVector, idx: Int): Matrix = {
    row match {
      case cv: CompIntDoubleVector =>
        val parts = cv.getPartitions
        MFactory.rbIntDoubleMatrix(
          (0 until row.getNumPartitions).filter(i => i != idx).toArray.map(i => parts(i))
        )
      case cv: CompIntFloatVector =>
        val parts = cv.getPartitions
        MFactory.rbIntFloatMatrix(
          (0 until row.getNumPartitions).filter(i => i != idx).toArray.map(i => parts(i))
        )
    }
  }
}
