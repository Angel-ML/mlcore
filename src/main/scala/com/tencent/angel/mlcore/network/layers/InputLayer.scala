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


abstract class InputLayer(name: String, outputDim: Int)(implicit graph: Graph)
  extends Layer(name, outputDim) {
  graph.addInputLayer(this)

  override def forward(): Matrix = {
    if (graph.isMatrixInCache(forwardKey)) {
      graph.getMatrixFromCache(forwardKey)
    } else {
      val forwardValue: Matrix = doForward(placeHolder.getFeats)
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
      doBackward(placeHolder.getFeats, gradInput)

      graph.put2Cache(backwardKey, null.asInstanceOf[Matrix])

      null.asInstanceOf[Matrix]
    }
  }

  protected def doBackward(input: Matrix, gradInput: Matrix): Unit
}
