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
package com.tencent.angel.mlcore.local.variables

import java.util.Random

import com.tencent.angel.ml.math2.matrix.Matrix
import com.tencent.angel.ml.math2.storage.{IntDoubleDenseVectorStorage, IntFloatDenseVectorStorage}
import com.tencent.angel.ml.math2.utils.RowType
import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.ml.math2.{MFactory, StorageType}
import com.tencent.angel.mlcore.conf.SharedConf
import com.tencent.angel.mlcore.network.EnvContext
import com.tencent.angel.mlcore.utils.{OptUtils, ValueNotAllowed}
import com.tencent.angel.mlcore.variable.{BlasMatVariable, Updater, VariableManager}


private[mlcore] class LocalBlasMatVariable(name: String,
                                           val numRows: Int,
                                           val numCols: Long,
                                           updater: Updater,
                                           rowType: RowType,
                                           formatClassName: String,
                                           allowPullWithIndex: Boolean)
                                          (implicit conf: SharedConf, variableManager: VariableManager)
  extends LocalVariable(name, rowType, updater, formatClassName, allowPullWithIndex) with BlasMatVariable {
  override protected var matrix: Matrix = _

  protected override def doCreate[T](envCtx: EnvContext[T]): Unit = {
    assert(envCtx == null || envCtx.client == null)
    storage = conf.valueType() match {
      case "float" =>
        MFactory.rbIntFloatMatrix(numSlot + 1, (numRows * numCols).toInt, StorageType.DENSE)
      case "double" =>
        MFactory.rbIntDoubleMatrix(numSlot + 1, (numRows * numCols).toInt, StorageType.DENSE)
      case _ => throw ValueNotAllowed("Value Not Allowed, Only Float/Double Are Allowed!")
    }
  }

  protected override def doInit(taskFlag: Int): Unit = {
    if (taskFlag == 0 && rowType.isDense) {
      val random = new Random()
      storage.getRow(0).getStorage match {
        case s: IntDoubleDenseVectorStorage =>
          val values = s.getValues
          values.indices.foreach { idx =>
            values(idx) = random.nextDouble() * stddev + mean
          }
        case s: IntFloatDenseVectorStorage =>
          val values = s.getValues
          values.indices.foreach { idx =>
            values(idx) = (random.nextDouble() * stddev + mean).toFloat
          }
        case _ =>
      }
    }
  }

  protected override def doPull(epoch: Int, indices: Vector = null): Unit = {
    if (matrix == null) {
      matrix = storage.getRow(0).getStorage match {
        case s: IntDoubleDenseVectorStorage =>
          MFactory.denseDoubleMatrix(numRows, numCols.toInt, s.getValues)
        case s: IntFloatDenseVectorStorage =>
          MFactory.denseFloatMatrix(numRows, numCols.toInt, s.getValues)
        case _ => throw ValueNotAllowed("Value Not Allowed, Only Float/Double Are Allowed!")
      }
    }
  }

  protected override def doPush(grad: Matrix, alpha: Double): Unit = {
    if (numSlot == 0) {
      OptUtils.getRowAsMatrix(storage, numSlot, numRows, numCols.toInt).isub(grad.imul(alpha))
    } else {
      OptUtils.getRowAsMatrix(storage, numSlot, numRows, numCols.toInt).iadd(grad)
    }
  }

  protected override def doRelease[T](envCtx: EnvContext[T]): Unit = {
    assert(envCtx == null || envCtx.client == null)
    storage = conf.valueType() match {
      case "float" =>
        MFactory.rbIntFloatMatrix(numSlot + 1, (numRows * numCols).toInt, StorageType.DENSE)
      case "double" =>
        MFactory.rbIntDoubleMatrix(numSlot + 1, (numRows * numCols).toInt, StorageType.DENSE)
      case _ => throw ValueNotAllowed("Value Not Allowed, Only Float/Double Are Allowed!")
    }
  }
}
