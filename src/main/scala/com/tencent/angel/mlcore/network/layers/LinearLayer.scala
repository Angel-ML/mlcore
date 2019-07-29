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
