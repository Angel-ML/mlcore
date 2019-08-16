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

import java.util
import java.util.concurrent.locks.ReentrantReadWriteLock

import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.ml.math2.matrix.Matrix

class DataCache() {
  private val matCache = new util.HashMap[String, Matrix]()
  private val vecCache = new util.HashMap[String, Vector]()

  private val lock = new ReentrantReadWriteLock()
  private val readLock = lock.readLock()
  private val writeLock = lock.writeLock()

  def addMatrix(name:String, mat: Matrix): Unit = {
    writeLock.lock()
    try {
      matCache.put(name, mat)
    } finally {
      writeLock.unlock()
    }
  }

  def addVector(name:String, vec: Vector): Unit = {
    writeLock.lock()
    try {
      vecCache.put(name, vec)
    } finally {
      writeLock.unlock()
    }
  }

  def hasMatrix(name:String): Boolean = {
    readLock.lock()

    try {
      matCache.containsKey(name)
    } finally {
      readLock.unlock()
    }
  }

  def hasVector(name:String): Boolean = {
    readLock.lock()

    try {
      vecCache.containsKey(name)
    } finally {
      readLock.unlock()
    }
  }

  def getMatrix(name: String): Matrix = {
    readLock.lock()

    try{
      matCache.getOrDefault(name, null.asInstanceOf[Matrix])
    } finally {
      readLock.unlock()
    }
  }

  def getVector(name: String): Vector = {
    readLock.lock()

    try{
      vecCache.getOrDefault(name, null.asInstanceOf[Vector])
    } finally {
      readLock.unlock()
    }
  }

  def clearAll(): Unit = {
    writeLock.lock()
    try {
      matCache.clear()
      vecCache.clear()
    } finally {
      writeLock.unlock()
    }
  }

  def clearMatrix(): Unit = {
    writeLock.lock()
    try {
      matCache.clear()
    } finally {
      writeLock.unlock()
    }
  }

  def clearVector(): Unit = {
    writeLock.lock()
    try {
      vecCache.clear()
    } finally {
      writeLock.unlock()
    }
  }
}
