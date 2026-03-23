"""在本地运行 Python 脚本的工具，用于生成 pickle 反序列化等 CTF payload。"""
import os
import subprocess
import tempfile

from langchain_core.tools import tool

# 单次输出最大字符数，防止刷屏
MAX_OUTPUT_CHARS = 50_000


@tool
def python_run(
    code: str,
    timeout: int = 10,
    python_binary: str = "python",
) -> str:
    """
    在本地执行一段 Python 代码并返回标准输出，常用于生成 pickle 反序列化 payload。

    参数:
    - code: 要执行的 Python 代码（完整脚本，可多行）。例如定义带 __reduce__ 的类并用 pickle.dumps 序列化后 print。
    - timeout: 最大执行秒数，默认 10
    - python_binary: Python 解释器路径，默认 "python"（可用 python3）

    返回:
    - 成功时返回脚本的 stdout（超出部分会截断说明）
    - 失败时返回 stderr 或异常信息

    典型用法（pickle 反序列化）:
    - 定义恶意类，实现 __reduce__ 返回 (os.system, ('id',)) 或 (eval, ('...',))
    - 使用 pickle.dumps(obj) 生成 payload，再 base64.b64encode 后 print，用于提交给题目
    """
    code = (code or "").strip()
    if not code:
        return "python_run: code 不能为空"

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".py",
            encoding="utf-8",
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            proc = subprocess.run(
                [python_binary, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.dirname(tmp_path),
            )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "<无输出>").strip()
            return f"python_run 退出码 {proc.returncode}:\n{err[:MAX_OUTPUT_CHARS]}"

        out = (proc.stdout or "").strip()
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n\n... (输出已截断，共 {len(proc.stdout)} 字符)"
        return out or "(无标准输出)"

    except FileNotFoundError:
        return f"未找到 Python 可执行文件: {python_binary}"
    except subprocess.TimeoutExpired:
        return f"python_run 执行超时（{timeout}s）"
    except Exception as e:
        return f"python_run 执行失败: {e}"
