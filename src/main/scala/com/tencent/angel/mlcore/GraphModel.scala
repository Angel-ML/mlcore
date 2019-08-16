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
package com.tencent.angel.mlcore

import com.tencent.angel.ml.math2.matrix.Matrix
import com.tencent.angel.mlcore.conf.SharedConf
import com.tencent.angel.mlcore.network.Graph
import com.tencent.angel.mlcore.optimizer.loss.LossFunc
import com.tencent.angel.mlcore.utils.{JsonUtils, MethodNotImplement}
import com.tencent.angel.mlcore.variable.Variable
import org.json4s.JsonAST.{JField, JObject, JString}
import org.json4s.jackson.JsonMethods.parse


abstract class GraphModel(conf: SharedConf) extends MLModel(conf) {
  val graph: Graph

  def updateConf(newConf: SharedConf): this.type = {
    newConf.allKeys().foreach{ key =>
      conf.set(key, newConf.get(key))
    }

    if (newConf.getJson != null) {
      conf.setJson(newConf.getJson)
    }

    this
  }

  def buildNetwork(): this.type

  override def addVariable(variable: Variable): this.type = {
    throw MethodNotImplement("addVariable is not implement in GraphModel")

    this
  }

  override def putSlot(v: Variable, g: Matrix): this.type = {
    throw MethodNotImplement("addVariable is not implement in GraphModel")
    this
  }

  def pushGradient(lr: Double): Unit = {
    pushSlot(lr)
  }

  def lossFunc: LossFunc = graph.getLossFunc
}

object GraphModel {

  def apply(className: String, conf: SharedConf): GraphModel = {
    val cls = Class.forName(className)
    val cstr = cls.getConstructor(classOf[SharedConf])
    cstr.newInstance(conf).asInstanceOf[GraphModel]
  }
}
