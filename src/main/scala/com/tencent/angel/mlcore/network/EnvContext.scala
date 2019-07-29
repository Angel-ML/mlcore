package com.tencent.angel.mlcore.network

trait EnvContext[T] {
  def client: T
}