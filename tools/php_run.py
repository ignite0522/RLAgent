"""在本地调用 PHP CLI 运行表达式并返回序列化字符串的工具。"""
import subprocess
import tempfile
import os
from langchain_core.tools import tool


@tool
def php_run(
    expr: str,
    php_binary: str = "php",
    timeout: int = 5,
) -> str:
    """
    使用本地 PHP 运行一小段代码并返回序列化结果,注意要url编码。

    参数:
    - expr: 一个合法的 PHP 表达式（不需要 <?php ?> 标签），例如 `"\"hello\""`, `array(\"a\"=>\"b\")` 或 `new MyClass()`
            注意：如果表达式中包含双引号，请确保在调用时正确转义或使用单引号包裹外层字符串。
    - php_binary: PHP 可执行文件路径，默认 `"php"`
    - timeout: 最大运行秒数

    返回:
    - 成功时返回 PHP 输出的序列化字符串
    - 失败时返回错误信息
    """
    try:
        expr_str = expr.strip()

        # 如果是包含 PHP 标签，移除它们
        if expr_str.startswith("<?"):
            # 去掉开头的 <?php 或 <? 并去掉可能的结束标签
            expr_str = expr_str
            expr_str = expr_str.replace("<?php", "", 1).replace("<?", "", 1)
            if "?>" in expr_str:
                expr_str = expr_str.split("?>", 1)[0]

        # 如果用户已经提供了完整的 PHP 语句（包含 echo/serialize/class/; 等），直接执行其内容
        needs_raw = (
            expr_str.startswith("echo")
            or "serialize(" in expr_str
            or "class " in expr_str
            or ";" in expr_str
            or "return " in expr_str
        )

        if needs_raw:
            code = expr_str
        else:
            # 仅是一个表达式，包装为 serialize(...)
            code = f'echo serialize({expr_str});'

        # 为避免 -r 在某些复杂代码或引号情况下产生解析问题，改为写入临时文件再执行。
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".php") as tf:
            # 确保以 PHP 标签开始
            content = code
            if "<?" not in content:
                content = "<?php\n" + content + "\n"
            tf.write(content)
            tmp_path = tf.name

        try:
            proc = subprocess.run([php_binary, tmp_path], capture_output=True, text=True, timeout=timeout)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        if proc.returncode != 0:
            stderr = proc.stderr.strip() if proc.stderr else "<no stderr>"
            return f"PHP 返回非零状态码 {proc.returncode}: {stderr}"
        return proc.stdout.strip()
    except FileNotFoundError:
        return f"未找到 PHP 可执行文件: {php_binary}"
    except subprocess.TimeoutExpired:
        return "PHP 执行超时"
    except Exception as e:
        return f"执行失败: {str(e)}"

