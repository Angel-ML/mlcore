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
package com.tencent.angel.mlcore.local

import com.tencent.angel.ml.math2.utils.LabeledData
import com.tencent.angel.mlcore.{GraphModel, PredictResult}
import com.tencent.angel.mlcore.conf.SharedConf
import com.tencent.angel.ml.math2.utils.DataBlock
import com.tencent.angel.mlcore.network.Graph
import com.tencent.angel.mlcore.utils.JsonUtils
import com.tencent.angel.mlcore.variable.{VariableManager, VariableProvider}


class LocalModel(conf: SharedConf) extends GraphModel(conf) {
  private implicit val sharedConf: SharedConf = conf

  override protected implicit val variableManager: VariableManager = new LocalVariableManager(isSparseFormat, conf)
  override protected val variableProvider: VariableProvider = new LocalVariableProvider(dataFormat, modelType)

  override implicit val graph: Graph = new Graph(variableProvider, conf, 1)

  override def buildNetwork(): this.type = {
    JsonUtils.layerFromJson(conf.getJson)

    this
  }

  def predict(storage: DataBlock[LabeledData]): List[PredictResult] = {
    graph.feedData((0 until storage.size()).toArray.map { idx => storage.get(idx) })
    pullParams(-1)
    graph.predict()
  }

  def predict(storage: LabeledData): PredictResult = {
    graph.feedData(Array(storage))
    pullParams(-1)
    graph.predict().head
  }
}
