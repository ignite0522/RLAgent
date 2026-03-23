# PHP 反序列化：常见防御与绕过

## 1. 常见防御

- 不反序列化用户输入；使用 `json_decode` 等替代 `unserialize`
- 过滤/禁用危险魔术方法（`__destruct`、`__wakeup`、`__toString` 等）
- 签名或 HMAC 校验序列化数据
- 禁止/限制可反序列化的类（白名单）

## 2. 利用思路

### 魔术方法链（POP Chain）

- **__destruct**：对象销毁时调用，常用于链入口
- **__wakeup**：unserialize 时先于 __destruct 调用；可尝试修改属性数量绕过 `__wakeup`（CVE-2016-7124：`O:4:"Demo":1:` 改为 `O:4:"Demo":2:`）
- **__toString**：对象被当作字符串使用时（如 echo、拼接）
- **__get / __set**：访问不可访问属性时
- **__call / __callStatic**：调用不可访问方法时
- **__invoke**：对象被当作函数调用时
- **__isset / __unset**：isset、unset 时
- **__sleep / __wakeup**：serialize / unserialize 时
- **__clone**：clone 时
- **__debugInfo**：var_dump 时

### 常见危险类

- **File 类**：通过 `__toString` 等读文件
- **SoapClient / SimpleXMLElement**：SSRF、XXE
- **Phar**：phar:// 触发反序列化，可结合文件上传
- **内置类**：`SplFileObject`、`DirectoryIterator`、`GlobIterator` 等读目录/文件

### 绕过

- **__wakeup 绕过**：属性数不匹配（PHP < 7.0.10 等）
- **私有/保护属性**：序列化字符串中私有属性为 `\0类名\0属性名`，保护为 `\0*\0属性名`，按实际长度构造
- **字符逃逸**：过滤导致长度变化时，通过多出的字符“吃掉”后面部分，改变解析边界
- **Phar**：上传 phar 文件（可伪造成 jpg 等），用 `phar://` 触发
