package com.tencent.angel.mlcore.local

import com.tencent.angel.mlcore.network.EnvContext

private [mlcore] case class NullClient()

private [mlcore] case class LocalEnvContext(override val client: NullClient = null) extends EnvContext[NullClient]
