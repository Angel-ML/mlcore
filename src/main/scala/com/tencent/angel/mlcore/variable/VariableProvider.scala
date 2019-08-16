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
package com.tencent.angel.mlcore.variable

import com.tencent.angel.mlcore.conf.SharedConf
import com.tencent.angel.mlcore.network.PlaceHolder

abstract class VariableProvider(implicit val conf: SharedConf, val variableManager: VariableManager) {
  def getMatVariable(name: String, numRows: Long, numCols: Long, updater: Updater,
                     formatClassName: String, allowPullWithIndex: Boolean): MatVariable

  def getEmbedVariable(name: String, numRows: Long, numCols: Long, updater: Updater,
                       formatClassName: String, placeHolder: PlaceHolder, taskNum: Int): EmbedVariable

  def getVecVariable(name: String, length: Long, updater: Updater, formatClassName: String,
                     allowPullWithIndex: Boolean): VecVariable
}
