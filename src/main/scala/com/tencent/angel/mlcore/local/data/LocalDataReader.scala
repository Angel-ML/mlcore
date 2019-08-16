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
package com.tencent.angel.mlcore.local.data


import com.tencent.angel.ml.math2.utils.{DataBlock, LabeledData}
import com.tencent.angel.mlcore.conf.SharedConf
import com.tencent.angel.mlcore.data.DataReader.BlockType.BlockType
import com.tencent.angel.mlcore.data.DataReader

import scala.io.Source

private[mlcore] class LocalDataReader(conf: SharedConf) extends DataReader(conf) {
  def sourceIter(pathName: String): Iterator[String] = {
    val source = Source.fromFile(pathName, "UTF-8")
    source.getLines()
  }

  override def getDataBlock(dbType: BlockType): DataBlock[LabeledData] = {
    new LocalMemoryDataBlock(-1, 1000 * 1024 * 1024)
  }
}
