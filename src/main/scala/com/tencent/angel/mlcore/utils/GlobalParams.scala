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

import org.json4s.{DefaultFormats, JField}
import org.json4s.JsonAST.{JBool, JDouble, JInt, JNothing, JObject, JString, JValue}
import com.tencent.angel.mlcore.utils.JsonUtils.extract
import com.tencent.angel.mlcore.conf.{MLCoreConf, SharedConf}

import scala.collection.mutable.ListBuffer

object GlobalKeys {
  val defaultOptimizer: String = "default_optimizer"
  val defaultLossFunc: String = "default_lossfunc"
  val lr: String = "lr"
  val data: String = "data"
  val model: String = "model"
  val train: String = "train"
  val path: String = "path"
  val format: String = "format"
  val indexRange: String = "indexrange"
  val numField: String = "numfield"
  val validateRatio: String = "validateratio"
  val sampleRatio: String = "sampleratio"
  val useShuffle: String = "useshuffle"
  val posnegRatio: String = "posnegratio"
  val transLabel: String = "translabel"
  val numClass: String = "numclass"
  val loadPath: String = "loadpath"
  val savePath: String = "savepath"
  val modelType: String = "modeltype"
  val modelSize: String = "modelsize"
  val blockSize: String = "blockSize"
  val epoch: String = "epoch"
  val numUpdatePerEpoch: String = "numupdateperepoch"
  val batchSize: String = "batchsize"
  val decay: String = "decay"
  val c: String = "kmeansc"
}


class DataParams(val path: Option[String],
                 val format: Option[String],
                 val indexRange: Option[Long],
                 val numField: Option[Int],
                 val validateRatio: Option[Double],
                 val sampleRatio: Option[Double],
                 val useShuffle: Option[Boolean],
                 val posnegRatio: Option[Double],
                 val transLabel: Option[String],
                 val numclass: Option[Int]
                ) {
  def updateConf(conf: SharedConf): Unit = {
    path.foreach(v => conf.set(MLCoreConf.ML_TRAIN_DATA_PATH, v))
    format.foreach(v => conf.set(MLCoreConf.ML_DATA_INPUT_FORMAT, v))
    indexRange.foreach(v => conf.setLong(MLCoreConf.ML_FEATURE_INDEX_RANGE, v))
    numField.foreach(v => conf.setInt(MLCoreConf.ML_FIELD_NUM, v))
    validateRatio.foreach(v => conf.setDouble(MLCoreConf.ML_VALIDATE_RATIO, v))
    sampleRatio.foreach(v => conf.setDouble(MLCoreConf.ML_BATCH_SAMPLE_RATIO, v))
    useShuffle.foreach(v => conf.setBoolean(MLCoreConf.ML_DATA_USE_SHUFFLE, v))
    posnegRatio.foreach(v => conf.setDouble(MLCoreConf.ML_DATA_POSNEG_RATIO, v))
    transLabel.foreach(v => conf.setString(MLCoreConf.ML_DATA_LABEL_TRANS, v))
    numclass.foreach(v => conf.setInt(MLCoreConf.ML_NUM_CLASS, v))
  }
}

object DataParams {
  implicit val formats = DefaultFormats

  private[mlcore] def apply(json: JValue): DataParams = {
    json match {
      case JNothing => new DataParams(None, None, None, None, None, None, None, None, None, None)
      case jast: JValue =>
        new DataParams(
          extract[String](jast, GlobalKeys.path),
          extract[String](jast, GlobalKeys.format),
          extract[Long](jast, GlobalKeys.indexRange),
          extract[Int](jast, GlobalKeys.numField),
          extract[Double](jast, GlobalKeys.validateRatio),
          extract[Double](jast, GlobalKeys.sampleRatio),
          extract[Boolean](jast, GlobalKeys.useShuffle),
          extract[Double](jast, GlobalKeys.posnegRatio),
          extract[String](jast, GlobalKeys.transLabel),
          extract[Int](jast, GlobalKeys.numClass)
        )
    }
  }

  private[mlcore] def toJson(conf: SharedConf): JObject = {
    val buf = ListBuffer[JField]()

    if (conf.hasKey(MLCoreConf.ML_TRAIN_DATA_PATH)) {
      val path = conf.get(MLCoreConf.ML_TRAIN_DATA_PATH)
      buf.append(JField(GlobalKeys.path, JString(path)))
    }

    if (conf.hasKey(MLCoreConf.ML_DATA_INPUT_FORMAT)) {
      val format = conf.get(MLCoreConf.ML_DATA_INPUT_FORMAT)
      buf.append(JField(GlobalKeys.format, JString(format)))
    }

    if (conf.hasKey(MLCoreConf.ML_FEATURE_INDEX_RANGE)) {
      val indexRange = conf.getLong(MLCoreConf.ML_FEATURE_INDEX_RANGE)
      buf.append(JField(GlobalKeys.indexRange, JInt(indexRange)))
    }

    if (conf.hasKey(MLCoreConf.ML_FIELD_NUM)) {
      val numField = conf.getInt(MLCoreConf.ML_FIELD_NUM)
      buf.append(JField(GlobalKeys.numField, JInt(numField)))
    }

    if (conf.hasKey(MLCoreConf.ML_VALIDATE_RATIO)) {
      val validateRatio = conf.getDouble(MLCoreConf.ML_VALIDATE_RATIO)
      buf.append(JField(GlobalKeys.validateRatio, JDouble(validateRatio)))
    }

    if (conf.hasKey(MLCoreConf.ML_BATCH_SAMPLE_RATIO)) {
      val sampleRatio = conf.getDouble(MLCoreConf.ML_BATCH_SAMPLE_RATIO)
      buf.append(JField(GlobalKeys.sampleRatio, JDouble(sampleRatio)))
    }

    if (conf.hasKey(MLCoreConf.ML_DATA_USE_SHUFFLE)) {
      val useShuffle = conf.getBoolean(MLCoreConf.ML_DATA_USE_SHUFFLE)
      buf.append(JField(GlobalKeys.useShuffle, JBool(useShuffle)))
    }

    if (conf.hasKey(MLCoreConf.ML_DATA_POSNEG_RATIO)) {
      val posnegRatio = conf.getDouble(MLCoreConf.ML_DATA_POSNEG_RATIO)
      buf.append(JField(GlobalKeys.posnegRatio, JDouble(posnegRatio)))
    }

    if (conf.hasKey(MLCoreConf.ML_DATA_LABEL_TRANS)) {
      val transLabel = conf.getString(MLCoreConf.ML_DATA_LABEL_TRANS)
      buf.append(JField(GlobalKeys.transLabel, JString(transLabel)))
    }

    if (conf.hasKey(MLCoreConf.ML_NUM_CLASS)) {
      val numclass = conf.getInt(MLCoreConf.ML_NUM_CLASS)
      buf.append(JField(GlobalKeys.numClass, JInt(numclass)))
    }

    JObject(buf.toList)
  }
}

class TrainParams(val epoch: Option[Int],
                  val numUpdatePerEpoch: Option[Int],
                  val batchSize: Option[Int],
                  val lr: Option[Double],
                  val decay: Option[Double],
                  val c: Option[Double]) {
  def updateConf(conf: SharedConf): Unit = {
    epoch.foreach(v => conf.setInt(MLCoreConf.ML_EPOCH_NUM, v))
    numUpdatePerEpoch.foreach(v => conf.setInt(MLCoreConf.ML_NUM_UPDATE_PER_EPOCH, v))
    batchSize.foreach(v => conf.setInt(MLCoreConf.ML_MINIBATCH_SIZE, v))
    lr.foreach(v => conf.setDouble(MLCoreConf.ML_LEARN_RATE, v))
    decay.foreach(v => conf.setDouble(MLCoreConf.ML_LEARN_DECAY, v))
    c.foreach(v => conf.setDouble(MLCoreConf.KMEANS_C, v))
  }
}

object TrainParams {
  implicit val formats = DefaultFormats

  private[mlcore] def apply(json: JValue): TrainParams = {
    json match {
      case JNothing => new TrainParams(None, None, None, None, None, None)
      case jast: JValue =>
        new TrainParams(
          extract[Int](jast, GlobalKeys.epoch),
          extract[Int](jast, GlobalKeys.numUpdatePerEpoch),
          extract[Int](jast, GlobalKeys.batchSize),
          extract[Double](jast, GlobalKeys.lr),
          extract[Double](jast, GlobalKeys.decay),
          extract[Double](jast, GlobalKeys.c)
        )
    }
  }

  private[mlcore] def toJson(conf: SharedConf): JObject = {
    val buf = ListBuffer[JField]()

    if (conf.hasKey(MLCoreConf.ML_EPOCH_NUM)) {
      val epoch = conf.getInt(MLCoreConf.ML_EPOCH_NUM)
      buf.append(JField(GlobalKeys.epoch, JInt(epoch)))
    }

    if (conf.hasKey(MLCoreConf.ML_NUM_UPDATE_PER_EPOCH)) {
      val numUpdatePerEpoch = conf.getInt(MLCoreConf.ML_NUM_UPDATE_PER_EPOCH)
      buf.append(JField(GlobalKeys.numUpdatePerEpoch, JInt(numUpdatePerEpoch)))
    }

    if (conf.hasKey(MLCoreConf.ML_MINIBATCH_SIZE)) {
      val batchSize = conf.getInt(MLCoreConf.ML_MINIBATCH_SIZE)
      buf.append(JField(GlobalKeys.batchSize, JInt(batchSize)))
    }

    if (conf.hasKey(MLCoreConf.ML_LEARN_RATE)) {
      val lr = conf.getDouble(MLCoreConf.ML_LEARN_RATE)
      buf.append(JField(GlobalKeys.lr, JDouble(lr)))
    }

    if (conf.hasKey(MLCoreConf.ML_LEARN_DECAY)) {
      val decay = conf.getDouble(MLCoreConf.ML_LEARN_DECAY)
      buf.append(JField(GlobalKeys.decay, JDouble(decay)))
    }

    if (conf.hasKey(MLCoreConf.KMEANS_C)) {
      val c = conf.getDouble(MLCoreConf.KMEANS_C)
      buf.append(JField(GlobalKeys.c, JDouble(c)))
    }

    JObject(buf.toList)
  }
}

class ModelParams(val loadPath: Option[String],
                  val savePath: Option[String],
                  val modelType: Option[String],
                  val modelSize: Option[Long],
                  val blockSize: Option[Int]) {
  def updateConf(conf: SharedConf): Unit = {
    loadPath.foreach(v => conf.set(MLCoreConf.ML_LOAD_MODEL_PATH, v))
    savePath.foreach(v => conf.set(MLCoreConf.ML_SAVE_MODEL_PATH, v))
    modelType.foreach(v => conf.set(MLCoreConf.ML_MODEL_TYPE, v))
    modelSize.foreach(v => conf.setLong(MLCoreConf.ML_MODEL_SIZE, v))
    blockSize.foreach(v => conf.setInt(MLCoreConf.ML_BLOCK_SIZE, v))
  }
}

object ModelParams {
  implicit val formats = DefaultFormats

  private[mlcore] def apply(json: JValue): ModelParams = {
    json match {
      case JNothing => new ModelParams(None, None, None, None, None)
      case jast: JValue =>
        new ModelParams(
          extract[String](jast, GlobalKeys.loadPath),
          extract[String](jast, GlobalKeys.savePath),
          extract[String](jast, GlobalKeys.modelType),
          extract[Long](jast, GlobalKeys.modelSize),
          extract[Int](jast, GlobalKeys.blockSize)
        )
    }
  }

  private[mlcore] def toJson(conf: SharedConf): JObject = {
    val buf = ListBuffer[JField]()

    if (conf.hasKey(MLCoreConf.ML_LOAD_MODEL_PATH)) {
      val loadPath = conf.getString(MLCoreConf.ML_LOAD_MODEL_PATH)
      buf.append(JField(GlobalKeys.loadPath, JString(loadPath)))
    }

    if (conf.hasKey(MLCoreConf.ML_SAVE_MODEL_PATH)) {
      val savePath = conf.getString(MLCoreConf.ML_SAVE_MODEL_PATH)
      buf.append(JField(GlobalKeys.savePath, JString(savePath)))
    }

    if (conf.hasKey(MLCoreConf.ML_MODEL_TYPE)) {
      val modelType = conf.getString(MLCoreConf.ML_MODEL_TYPE)
      buf.append(JField(GlobalKeys.modelType, JString(modelType)))
    }

    if (conf.hasKey(MLCoreConf.ML_MODEL_SIZE)) {
      val modelSize = conf.getLong(MLCoreConf.ML_MODEL_SIZE)
      buf.append(JField(GlobalKeys.modelSize, JInt(modelSize)))
    }

    if (conf.hasKey(MLCoreConf.ML_BLOCK_SIZE)) {
      val blockSize = conf.getInt(MLCoreConf.ML_BLOCK_SIZE)
      buf.append(JField(GlobalKeys.blockSize, JInt(blockSize)))
    }

    JObject(buf.toList)
  }

}
