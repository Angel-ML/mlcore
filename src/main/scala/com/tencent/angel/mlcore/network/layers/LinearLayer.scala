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
import com.tencent.angel.mlcore.network.Graph
import com.tencent.angel.mlcore.utils.LayerKeys
import org.json4s.JsonAST.{JField, JString}
import org.json4s.JsonDSL._


abstract class LinearLayer(name: String, outputDim: Int, val inputLayer: Layer)(implicit graph: Graph)
  extends Layer(name, outputDim) {
  this.addInput(inputLayer)
  inputLayer.addConsumer(this)

  override def forward(): Matrix = {
    if (graph.isMatrixInCache(forwardKey)) {
      graph.getMatrixFromCache(forwardKey)
    } else {
      val forwardValue: Matrix = doForward(inputLayer.forward())
      graph.put2Cache(forwardKey, forwardValue)
      forwardValue
    }
  }

  protected def doForward(input: Matrix): Matrix

  override def backward(layer: Layer): Matrix = {
    if (graph.isMatrixInCache(backwardKey)) {
      graph.getMatrixFromCache(backwardKey)
    } else {
      val gradInput = gatherGradInput()
      val backwardValue = doBackward(inputLayer.forward(), gradInput)

      graph.put2Cache(backwardKey, backwardValue)

      backwardValue
    }
  }

  protected def doBackward(input: Matrix, gradInput: Matrix): Matrix

  override def toString: String = {
    s"${this.getClass.getSimpleName} name=$name outputDim=$outputDim inputLayer=${inputLayer.name} "
  }

  private[mlcore] override def toJson: JField = {
    val layerJson = (LayerKeys.typeKey -> s"${this.getClass.getSimpleName}") ~
      (LayerKeys.outputDimKey -> outputDim) ~
      (LayerKeys.inputLayerKey, JString(inputLayer.name))

    JField(name, layerJson)
  }
}
