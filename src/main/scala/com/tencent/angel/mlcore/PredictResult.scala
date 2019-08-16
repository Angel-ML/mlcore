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

class PredictResult(val sid: String, val pred: Double, val proba: Double, val predLabel: Double,
                    val trueLabel: Double, val attached: Double = Double.NaN) {
  var separator = ", "

  def getText: String = {
    val predStr = if (pred == Double.NaN) "" else f"$pred%.2f$separator"
    val probaStr = if (proba == Double.NaN) "" else f"$proba%.2f$separator"
    val trueLabelStr = if (trueLabel == Double.NaN) "" else f"$trueLabel%.2f"
    f"$sid%s$separator$predLabel%.2f$separator$predStr$probaStr$trueLabelStr"
  }
}

object PredictResult {
  def apply(sid: String, pred: Double, proba: Double, predLabel: Double,
            trueLabel: Double, attached: Double = Double.NaN): PredictResult = {
    new PredictResult(sid, pred, proba, predLabel, trueLabel, attached)
  }
}
