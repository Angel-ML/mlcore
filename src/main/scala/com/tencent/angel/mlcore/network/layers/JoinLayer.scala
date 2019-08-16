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
import org.json4s.JsonAST.{JArray, JField, JString}
import org.json4s.JsonDSL._


abstract class JoinLayer(name: String, outputDim: Int, val inputLayers: Array[Layer])(implicit graph: Graph)
  extends Layer(name, outputDim) {
  inputLayers.foreach { layer =>
    layer.addConsumer(this)
    this.addInput(layer)
  }

  override def forward(): Matrix = {
    if (graph.isMatrixInCache(forwardKey)) {
      graph.getMatrixFromCache(forwardKey)
    } else {
      var usedNames = Set[String]()
      val inputs = inputLayers.map { layer =>
        //  layer.name -> layer.forward()
        val name = layer.name
        val newName = if (usedNames.contains(name)) {
          name + "_x"
        } else {
          usedNames += name
          name
        }
        newName -> layer.forward()
      }.toMap
      val forwardValue: Matrix = doForward(inputs)
      graph.put2Cache(forwardKey, forwardValue)
      forwardValue
    }
  }

  protected def doForward(inputs: Map[String, Matrix]): Matrix

  override def backward(layer: Layer): Matrix = {
    assert(isInput(layer))
    if (graph.isMatrixInCache(backwardKey)) {
      val key = s"$backwardKey/${layer.name}"
      graph.getMatrixFromCache(key)
    } else {
      val gradInput = gatherGradInput()
      var usedNames = Set[String]()
      val inputs = inputLayers.map { layer =>
        //        layer.name -> layer.forward()
        val name = layer.name
        val newName = if (usedNames.contains(name)) {
          name + "_x"
        } else {
          usedNames += name
          name
        }
        newName -> layer.forward()
      }.toMap
      val matrixMap = doBackward(inputs, gradInput)
      graph.put2Cache(backwardKey, null.asInstanceOf[Matrix])
      matrixMap.foreach { case (layerName: String, mat: Matrix) =>
        val key = s"$backwardKey/$layerName"
        graph.put2Cache(key, mat)
      }

      matrixMap(layer.name)
    }
  }

  protected def doBackward(inputs: Map[String, Matrix], gradInput: Matrix): Map[String, Matrix]

  private[mlcore] override def toJson: JField = {
    val layerJson = (LayerKeys.typeKey -> s"${this.getClass.getSimpleName}") ~
      (LayerKeys.outputDimKey -> outputDim) ~
      (LayerKeys.inputLayersKey -> JArray(inputLayers.toList.map(layer => JString(layer.name))))

    JField(name, layerJson)
  }


}
