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
package com.tencent.angel.mlcore.local.optimizer

import java.util.concurrent.Future

import com.tencent.angel.ml.math2.ufuncs.OptFuncs
import com.tencent.angel.mlcore.conf.{MLCoreConf, SharedConf}
import com.tencent.angel.mlcore.local.LocalOptimizerProvider
import com.tencent.angel.mlcore.local.variables.{LocalBlasMatVariable, LocalMatVariable, LocalVecVariable}
import com.tencent.angel.mlcore.optimizer.Optimizer
import com.tencent.angel.mlcore.utils.{OptUtils, OptimizerKeys}
import com.tencent.angel.mlcore.variable.Variable
import org.json4s.JsonAST.JObject
import org.json4s.JsonDSL._


private[mlcore] class Momentum(override var lr: Double, val momentum: Double) extends Optimizer {
  override val numSlot: Int = 2

  override def update[T](variable: Variable, epoch: Int, batchSize: Int): Future[T] = {
    variable match {
      case v: LocalBlasMatVariable =>
        val value = v.storage.getRow(0)
        val moment = v.storage.getRow(1)
        val grad = v.storage.getRow(2)
        //        val grad = if (regL2Param == 0) {
        //          v.storage.getRow(2)
        //        } else {
        //          value.axpy(v.storage.getRow(2), regL2Param)
        //        }

        OptFuncs.iexpsmoothing(moment, grad, momentum)
        value.isub(moment.mul(lr))
        //        if (regL1Param != 0.0) {
        //          Ufuncs.isoftthreshold(value, regL1Param)
        //        }

        grad.imul(0.0)
      case v: LocalMatVariable =>
        val numFactors: Int = v.numRows
        val value = OptUtils.getRowsAsMatrix(v.storage, 0, numFactors)
        val moment = OptUtils.getRowsAsMatrix(v.storage, numFactors, numFactors * 2)
        val grad = OptUtils.getRowsAsMatrix(v.storage, numFactors * 2, numFactors * 3)
        //        val grad = if (regL2Param == 0) {
        //          OptUtils.getRowsAsMatrix(v.storage, numFactors, numFactors * 3)
        //        } else {
        //          value.axpy(OptUtils.getRowsAsMatrix(v.storage, numFactors, numFactors * 3), regL2Param)
        //        }

        OptFuncs.iexpsmoothing(moment, grad, momentum)
        value.isub(moment.mul(lr))
        //        if (regL1Param != 0.0) {
        //          Ufuncs.isoftthreshold(value, regL1Param)
        //        }

        grad.imul(0.0)
      case v: LocalVecVariable =>
        val value = v.storage.getRow(0)
        val moment = v.storage.getRow(1)
        val grad = v.storage.getRow(2)
        //        val grad = if (regL2Param == 0) {
        //          v.storage.getRow(2)
        //        } else {
        //          value.axpy(v.storage.getRow(2), regL2Param)
        //        }

        OptFuncs.iexpsmoothing(moment, grad, momentum)
        value.isub(moment.mul(lr))
        //        if (regL1Param != 0.0) {
        //          Ufuncs.isoftthreshold(value, regL1Param)
        //        }

        grad.imul(0.0)
    }

    null.asInstanceOf[Future[T]]
  }
}

private[mlcore] object Momentum {

  private[mlcore] def fromJson(jast: JObject, provider: LocalOptimizerProvider)(implicit conf: SharedConf): Momentum = {
    val laProvider = provider.asInstanceOf[LocalOptimizerProvider]
    assert(laProvider.fieldEqualClassName[Momentum](jast, OptimizerKeys.typeKey))
    val moment = conf.getDouble(MLCoreConf.ML_OPT_MOMENTUM_MOMENTUM, MLCoreConf.DEFAULT_ML_OPT_MOMENTUM_MOMENTUM)

    val regL1Param: Double = conf.getDouble(MLCoreConf.ML_REG_L1, MLCoreConf.DEFAULT_ML_REG_L1)
    val regL2Param: Double = conf.getDouble(MLCoreConf.ML_REG_L2, MLCoreConf.DEFAULT_ML_REG_L2)
    val opt = new Momentum(1.0, laProvider.extract[Double](jast, OptimizerKeys.momentumKey, Some(moment)).get)
    opt.setRegL1Param(regL1Param).setRegL2Param(regL2Param)
  }
}
