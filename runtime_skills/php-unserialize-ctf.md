## PHP 反序列化（unserialize）CTF 操作指南（Skill）

### 目标

定位 `unserialize` 入口与可控参数，识别可用 POP 链（`__wakeup`/`__destruct`/`__toString`/`__invoke` 等），构造 payload 触发读文件/命令执行/写入，拿到 flag。

### 强制规则（必须遵守）

1) **严禁手工构造精确序列化字符串**

任何需要**精确 PHP 序列化结果**的场景（Cookie/GET/POST 参数/HTTP header 中的 serialize 字符串，或依赖精确长度、私有属性名 `\0Class\0prop` 等），必须调用 `php_run` 生成。

- 禁止：在模型输出或工具参数里手写/猜测 `O:...`、长度数字、私有/保护属性序列化前缀、字符串长度等。
- 允许：描述对象结构与构造思路，但最终 serialize 字符串必须由 `php_run` 产生。

2) **发送前必须 URL 编码（必要时双编码）**

从 `php_run` 得到 serialize 原始结果后：

- 必须进行 URL 编码（`urlencode` / `rawurlencode` / 等效 percent-encoding）。
- 若怀疑中间链路会解码一次（框架/反代/中间件），使用双重编码并说明原因。

3) **发送时只允许把“已编码”结果放进 `curl_request`**

把“已编码”的 payload 放入 `curl_request` 的 `cookies` / `params` / `headers` 字段中（并确保 `as_source=false`）。

不要在自然语言里输出未编码的 serialize 字符串，也不要试图手工修改其长度字段。

4) **说明可复现信息**

在你的文字说明里，必须明确：

- 是否做了 URL 编码/双重编码
- 放置位置（cookie / param / header）

### 推荐流程（按顺序）

1. `dirsearch_scan`：找入口/备份/源码泄露点（如 `.bak`、`~`、`.swp`、`phpinfo`、`source`）。
2. `curl_request`：读关键源码，确认（请设置 `as_source=true`）：
   - `unserialize` 调用点
   - 可控参数来源（Cookie/GET/POST/Header/Session）
   - 反序列化前后的过滤/解码（`urldecode`/`base64_decode`/签名校验/HMAC）
   - 类定义与魔术方法链是否存在/可控
3. 需要 payload：`php_run` 生成 `serialize(...)` 原始字符串。
4. URL 编码（必要时双编码），再 `curl_request` 发送并迭代（保持 `as_source=false`）。
5. 拿到 flag 后停止。

### 示例（仅示范格式）

- 第一步：用 `php_run` 生成 serialize 原始字符串
  - tool_call: `{"name":"php_run","args":{"expr":"serialize(...)"}}`
- 第二步：将上一步结果 URL 编码后，用 `curl_request` 发送
  - tool_call: `{"name":"curl_request","args":{"url":"<target>","method":"GET","as_source":false,"cookies":{"user":"<urlencode(serialize_result)>"}}}`

### 卡住时兜底（不要直接放弃）

先调用：`read_doc("php_deserialize")`，再继续上述流程。

### 常见过滤提醒

- 可能过滤 `cat`：可尝试 `tac` 或其它等效方式
- 可能屏蔽 `flag` 关键字：可尝试模糊匹配（如 `f*`）或换读取方式

