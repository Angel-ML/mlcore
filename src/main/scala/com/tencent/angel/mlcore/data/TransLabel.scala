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
package com.tencent.angel.mlcore.data

import com.tencent.angel.mlcore.utils.JsonUtils.matchClassName

sealed trait TransLabel extends Serializable {
  def trans(label: Double): Double
}

class NoTrans extends TransLabel{
  override def trans(label: Double): Double = label
}

class PosNegTrans(threshold: Double = 0) extends TransLabel{
  override def trans(label: Double): Double = {
    if (label > threshold) 1.0 else - 1.0
  }
}

class ZeroOneTrans(threshold: Double = 0) extends TransLabel{
  override def trans(label: Double): Double = {
    if (label > threshold) 1.0 else 0.0
  }
}

class AddOneTrans extends TransLabel {
  override def trans(label: Double): Double = label + 1.0
}

class SubOneTrans extends TransLabel {
  override def trans(label: Double): Double = label - 1.0
}


object TransLabel {
  def get(name: String, threshold: Double = 0): TransLabel = {
    name.toLowerCase match {
      case name: String if matchClassName[NoTrans](name) => new NoTrans()
      case name: String if matchClassName[PosNegTrans](name) => new PosNegTrans(threshold)
      case name: String if matchClassName[ZeroOneTrans](name) => new ZeroOneTrans(threshold)
      case name: String if matchClassName[AddOneTrans](name) => new AddOneTrans()
      case name: String if matchClassName[SubOneTrans](name) => new SubOneTrans()
    }
  }
}


