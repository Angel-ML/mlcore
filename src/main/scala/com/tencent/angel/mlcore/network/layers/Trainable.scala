package com.tencent.angel.mlcore.network.layers

import com.tencent.angel.mlcore.optimizer.Optimizer

trait Trainable {
  def optimizer: Optimizer
}