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

import java.lang.{Long => JLong}
import java.util.{HashMap => JHashMap, Map => JMap}

import com.tencent.angel.ml.math2.matrix.{MapMatrix, Matrix}
import com.tencent.angel.ml.math2.storage._
import com.tencent.angel.ml.math2.utils.RowType
import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.mlcore.conf.SharedConf
import com.tencent.angel.mlcore.network.{EnvContext, PlaceHolder}
import com.tencent.angel.mlcore.utils.ValueNotAllowed
import com.tencent.angel.mlcore.variable.{EmbedUtils, EmbedVariable, Updater, VariableManager}


private[mlcore] class LocalEmbedVariable(name: String,
                                         numRows: Int,
                                         numCols: Long,
                                         updater: Updater,
                                         rowType: RowType,
                                         formatClassName: String,
                                         allowPullWithIndex: Boolean,
                                         placeHolder: PlaceHolder)
                                        (implicit conf: SharedConf, variableManager: VariableManager)
  extends LocalMatVariable(name, numRows, numCols, updater, rowType, formatClassName, allowPullWithIndex) with EmbedVariable {
  private val embeddings: JMap[JLong, Vector] = new JHashMap[JLong, Vector]()

  protected override def doPull(epoch: Int, indices: Vector = null): Unit = {
    embeddings.clear()

    if (indices != null) {
      indices.getStorage match {
        case s: IntIntDenseVectorStorage =>
          s.getValues.foreach { idx => embeddings.put(idx.toLong, storage.getRow(idx)) }
        case s: IntLongDenseVectorStorage =>
          s.getValues.foreach { idx => embeddings.put(idx, storage.getRow(idx.toInt)) }
      }
    } else {
      (0 until numRows).foreach { idx =>
        embeddings.put(idx.toLong, storage.getRow(idx))
      }
    }

    val matStats = EmbedUtils.geneMatrix(placeHolder, assembleHint, embeddings)
    matrix = matStats._1
    assembleStats = matStats._2
  }

  protected override def doPush(grad: Matrix, alpha: Double): Unit = {
    if (numSlot == 0) {
      grad match {
        case mat: MapMatrix[_] =>
          val tempMap = mat.getMap
          val iter = tempMap.keySet().iterator()
          while (iter.hasNext) {
            val key = iter.next()
            val vector = tempMap.get(key)

            storage.getRow(key.toInt).isub(vector.imul(alpha))
          }
        case _ => throw ValueNotAllowed("Only MapMatrix is allow as a gradient matrix!")
      }
    } else {
      grad match {
        case mat: MapMatrix[_] =>
          val tempMap = mat.getMap
          val iter = tempMap.keySet().iterator()
          val start = numRows * numSlot
          while (iter.hasNext) {
            val key = iter.next()
            val vector = tempMap.get(key)

            storage.getRow(start + key.toInt).iadd(vector)
          }
        case _ => throw ValueNotAllowed("Only MapMatrix is allow as a gradient matrix!")
      }
    }
  }
}
