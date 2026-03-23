"""curl 请求工具。

由上层 LLM 通过 `curl_args` 给出 curl 的"额外选项字符串"（可选择是否包含开头的 `curl`）。
工具只负责执行 curl，并将响应的 headers/body 返回给上层。
"""

import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from langchain_core.tools import tool


# 匹配 HTTP 响应状态行，例如 "HTTP/1.1 200 OK"
_HTTP_STATUS_RE = re.compile(r"^HTTP/\d+(?:\.\d+)?\s+(\d+)\s+(.*)$", re.MULTILINE)


def _filter_output_args(ts: list[str]) -> list[str]:
    """
    过滤掉可能覆盖我们临时文件输出的参数，避免模型里出现 -D/-o/--output 等。
    """
    out: list[str] = []
    i = 0
    while i < len(ts):
        t = ts[i]
        # -D file / --dump-header file / --dump-header=file
        if t in ("-D", "--dump-header"):
            i += 2
            continue
        if t.startswith("--dump-header="):
            i += 1
            continue
        if t.startswith("-D") and t != "-D":
            i += 1
            continue
        # -o file / --output file / --output=file
        if t in ("-o", "--output"):
            i += 2
            continue
        if t.startswith("--output="):
            i += 1
            continue
        if t.startswith("-o") and t != "-o":
            i += 1
            continue
        # -O / --remote-name：会在当前目录写文件，不希望模型这么做
        if t in ("-O", "--remote-name"):
            i += 1
            continue
        out.append(t)
        i += 1
    return out


@tool
def curl_request(
    url: str,
    curl_args: str | None = None,
    as_source: bool = True,
    max_chars: int = 4000,
) -> str:
    """
    执行 curl 请求并返回结果。

    - `curl_args`：curl 的"额外选项字符串"（可带也可不带最前面的 `curl` 前缀）。
      具体传入的内容（例如 headers 的值、data 的 body、cookie 字符串、表单字段名等）由大模型决定。
      常用可选项（按需组合，均由大模型决定取值）：
      1) `-X <METHOD>`：指定方法（GET/POST/PUT/PATCH/DELETE...）
      2) `-H "<Header: Value>"`：增加请求头（可多次出现）
      3) `--data-raw <BODY>`：请求体（常用于 JSON/plain text，body 内容由大模型决定）
      4) `--data-urlencode <K>=<V>`：表单/查询编码（K/V 由大模型决定）
      5) `-F "<field>=@<path>;filename=<name>"`：multipart/form-data 上传（字段名/文件路径由大模型决定）
      6) `-b "<k=v; ...>"` 或 `--cookie "<k=v; ...>"`：Cookie（由大模型决定）
      7) `-u "<user:pass>"`：Basic Auth（由大模型决定）
      8) `--compressed`：接收并自动解压 gzip/deflate（由大模型决定是否启用）
      9) `--max-time <seconds>`：请求超时（由大模型决定是否启用）

      注意：工具会自己捕获响应头/响应体，因此不要在 `curl_args` 中加入下面这些"输出相关参数"（否则会被过滤掉）：
      `-D/--dump-header`、`-o/--output`、`-O/--remote-name`、`--output=...`。

    - `as_source=true`：返回 `文件内容:\\n...`
    - `as_source=false`：兼容旧调用；本工具始终按 `as_source=true` 返回“原文文件内容”，不再提供 JSON 摘要格式
    """
    curl_args = (curl_args or "").strip()
    tokens: list[str] = []
    if curl_args:
        tokens = shlex.split(curl_args)
        if tokens and tokens[0] == "curl":
            tokens = tokens[1:]
        tokens = _filter_output_args(tokens)

    # 检查 tokens 里是否已包含 URL（避免重复追加）
    has_url = any(tok.startswith("http://") or tok.startswith("https://") for tok in tokens)

    tmp_dir = Path(tempfile.mkdtemp(prefix="curl_request_"))
    headers_path = tmp_dir / "headers.txt"
    body_path = tmp_dir / "body.bin"

    try:
        # 基础命令：-sS 静默但显示错误，--max-redirs 0 不跟随重定向
        # -D 将响应头写入临时文件，-o 将响应体写入临时文件
        cmd: list[str] = [
            "curl",
            "-sS",
            "--max-redirs",
            "0",
            "-D",
            str(headers_path),
            "-o",
            str(body_path),
        ] + tokens
        if not has_url:
            cmd.append(url)

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            return f"请求执行失败: {err}"

        # 从响应头文件中解析状态码（取最后一个状态行，兼容重定向场景）
        headers_text = headers_path.read_text(encoding="utf-8", errors="replace")
        matches = list(_HTTP_STATUS_RE.finditer(headers_text))
        if matches:
            code = int(matches[-1].group(1))
            reason = matches[-1].group(2).strip()
        else:
            code = -1
            reason = "unknown"

        body_bytes = body_path.read_bytes()
        body_text = body_bytes.decode("utf-8", errors="replace")
        if max_chars is not None:
            body_text = body_text[:max_chars]

        # 不做任何“摘要/压缩”：直接返回响应体原文（仍会受到 max_chars 截断）
        # 同时为了兼容旧调用，无论 as_source 怎么传都返回文件内容。
        return f"文件内容:\n{body_text}"
    except Exception as e:
        return f"请求执行失败: {type(e).__name__}:{e}"
    finally:
        # 清理临时目录
        try:
            for p in tmp_dir.rglob("*"):
                try:
                    p.unlink(missing_ok=True)  # type: ignore[attr-defined]
                except Exception:
                    pass
            tmp_dir.rmdir()
        except Exception:
            pass