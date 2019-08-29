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
package com.tencent.angel.mlcore.network.layers


import com.tencent.angel.ml.math2.matrix.Matrix
import com.tencent.angel.mlcore.PredictResult
import com.tencent.angel.mlcore.network.Graph
import com.tencent.angel.mlcore.optimizer.loss.LossFunc
import com.tencent.angel.mlcore.utils.LayerKeys
import org.json4s.JsonAST.{JField, JString}
import org.json4s.JsonDSL._


class LossLayer(name: String, val inputLayer: Layer, val lossFunc: LossFunc)(implicit graph: Graph)
  extends Layer(name, -1) {
  inputLayer.addConsumer(this)
  this.addInput(inputLayer)
  graph.setLossLayer(this)
  lossFunc.setLossLayer(this)

  def getLabel: Matrix = {
    if (placeHolder == null) {
      throw new Exception("placeHolder is empty, pls. set first!")
    }
    placeHolder.getLabel
  }

  def getAttached: Array[String] = {
    if (placeHolder == null) {
      throw new Exception("placeHolder is empty, pls. set first!")
    }
    placeHolder.getAttached
  }

  def predict(): List[PredictResult] = {
    lossFunc.predict(forward(), graph)
  }

  def calLoss(): Double = {
    lossFunc.calLoss(forward(), graph)
  }

  override def forward(): Matrix = {
    if (graph.isMatrixInCache(forwardKey)) {
      graph.getMatrixFromCache(forwardKey)
    } else {
      val forwardValue: Matrix = inputLayer.forward()
      graph.put2Cache(forwardKey, forwardValue)
      forwardValue
    }
  }

  override def backward(layer: Layer): Matrix = {
    if (graph.isMatrixInCache(backwardKey)) {
      val inputs = getAllInputs
      if (inputs.isEmpty) {
        null.asInstanceOf[Matrix]
      } else if (inputs.size == 1) {
        val validateLayer = if (layer != null) {
          layer
        } else {
          getAllInputs.head
        }

        assert(isInput(validateLayer))
        val key = s"$backwardKey/${validateLayer.name}"
        graph.getMatrixFromCache(key)
      } else {
        assert(isInput(layer))
        val key = s"$backwardKey/${layer.name}"
        graph.getMatrixFromCache(key)
      }
    } else {
      val grad = lossFunc.calGrad(forward(), graph)
      graph.put2Cache(backwardKey, grad)

      grad
    }

  }

  private[mlcore] override def toJson: JField = {
    val layerJson = (LayerKeys.typeKey -> s"${this.getClass.getSimpleName}") ~
      (LayerKeys.outputDimKey -> outputDim) ~
      (LayerKeys.inputLayerKey, JString(inputLayer.name)) ~
      (LayerKeys.lossFuncKey, lossFunc.toJson)


    JField(name, layerJson)
  }
}
