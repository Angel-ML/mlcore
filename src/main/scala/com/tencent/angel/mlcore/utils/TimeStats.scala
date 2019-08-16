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
package com.tencent.angel.mlcore.utils

import org.apache.commons.logging.{Log, LogFactory}

class TimeStats(
                 var forwardTime: Long = 0,
                 var backwardTime: Long = 0,
                 var pullParamsTime: Long = 0,
                 var pushParamsTime: Long = 0,
                 var updateTime: Long = 0,
                 var createTime: Long = 0,
                 var initTime: Long = 0,
                 var loadTime: Long = 0,
                 var saveTime: Long = 0,
                 var predictTime: Long = 0
               ) extends Serializable {
  val LOG: Log = LogFactory.getLog(classOf[TimeStats])

  def summary(): String = {
    val summaryString = s"\nSummary: \n\t" +
      s"forwardTime = $forwardTime, \n\tbackwardTime = $backwardTime, \n\t" +
      s"pullParamsTime = $pullParamsTime, \n\tcreateTime = $createTime, \n\t" +
      s"initTime = $initTime, \n\tloadTime = $loadTime, \n\t" +
      s"saveTime = $saveTime, \n\tpredictTime = $predictTime, \n\t" +
      s"pushParamsTime = $pushParamsTime, \n\tupdateTime = $updateTime"

    LOG.info(summaryString)
    summaryString
  }
}