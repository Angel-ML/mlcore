package com.tencent.angel.mlcore

import com.tencent.angel.ml.math2.utils.LabeledData
import com.tencent.angel.mlcore.data.DataBlock

trait Predictor {
  def predict(storage: DataBlock[LabeledData]): DataBlock[PredictResult]

  def predict(storage: LabeledData): PredictResult
}
