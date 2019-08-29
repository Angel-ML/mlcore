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


package com.tencent.angel.mlcore.optimizer

import com.tencent.angel.mlcore.conf.SharedConf
import com.tencent.angel.mlcore.utils.OptimizerKeys
import com.tencent.angel.mlcore.variable.Updater
import org.json4s.JsonAST.{JField, JObject, JString}
import org.json4s.JsonDSL._

import scala.collection.mutable.ListBuffer

trait Optimizer extends Updater with Serializable {
  var lr: Double
  val epsilon: Double = 1e-10

  protected var regL1Param: Double = 0.0
  protected var regL2Param: Double = 0.0

  def setLR(lr: Double): this.type = {
    this.lr = lr
    this
  }

  def setRegL1Param(regParam: Double): this.type = {
    this.regL1Param = regParam
    this
  }

  def setRegL2Param(regParam: Double): this.type = {
    this.regL2Param = regParam
    this
  }

  def getLR: Double = this.lr

  def getRegL1Param: Double = this.regL1Param

  def getRegL2Param: Double = this.regL2Param
}

object Optimizer {

  def toJson(optimizer:Optimizer): JObject = {
    val buf = ListBuffer[JField]()

    val name = optimizer.getClass.getSimpleName
    buf.append(JField(OptimizerKeys.typeKey, JString(name)))

    val fields = List(
      OptimizerKeys.momentumKey,
      OptimizerKeys.alphaKey,
      OptimizerKeys.betaKey,
      OptimizerKeys.gammaKey
    )

    fields.foreach{ key =>
      try {
        val field = optimizer.getClass.getDeclaredField(key)
        field.setAccessible(true)
        val value = field.getDouble(optimizer)
        buf.append(JField(key, value))
      } catch {
        case e: NoSuchFieldException =>
      }
    }

    val getRegL1Param = optimizer.getClass.getMethod("getRegL1Param")
    val regL1 = getRegL1Param.invoke(optimizer).asInstanceOf[Double]
    if (regL1 != 0.0) {
      buf.append(JField(OptimizerKeys.reg1Key, regL1))
    }

    val getRegL2Param = optimizer.getClass.getMethod("getRegL2Param")
    val regL2 = getRegL2Param.invoke(optimizer).asInstanceOf[Double]
    if (regL1 != 0.0) {
      buf.append(JField(OptimizerKeys.reg2Key, regL2))
    }

    JObject(buf.toList)
  }

  def getOptimizerProvider(className: String, conf: SharedConf): OptimizerProvider = {
    val cls = Class.forName(className)
    val constructor = cls.getConstructor(classOf[SharedConf])
    constructor.newInstance(conf).asInstanceOf[OptimizerProvider]
  }
}



