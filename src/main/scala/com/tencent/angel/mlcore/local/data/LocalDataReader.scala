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
