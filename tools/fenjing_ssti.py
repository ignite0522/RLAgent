"""调用 FenJing 对目标 URL 做 Jinja2 SSTI 扫描（极简封装）。"""
import subprocess
from urllib.parse import urlparse, urlunparse

from langchain_core.tools import tool


def _normalize_url_for_fenjing(url: str) -> str:
    """
    规范 URL 供 FenJing 使用：
    - 去除查询参数（?xxx），让 FenJing 自己发现并 fuzz 参数
    - 对 ctf.show 等挑战域名优先使用 HTTPS
    """
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return url
    parsed = urlparse(url)
    #  strip query and fragment
    base = urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/") or "/", "", "", ""))
    # ctf.show 等挑战站点优先 HTTPS
    if "ctf.show" in parsed.netloc.lower() and parsed.scheme == "http":
        base = base.replace("http://", "https://", 1)
    return base


@tool
def fenjing_ssti(
    target_url: str,
    exec_cmd: str = "ls /",
    timeout: int = 300,
) -> str:
    """
    使用 FenJing 自动检测目标 URL 是否存在 Jinja2 模板注入 (SSTI) 漏洞。
    传入基础 URL（不含查询参数）即可，FenJing 会自动发现并 fuzz 参数。
    本工具会自动：去除 URL 中的 ?xxx 查询串；对 ctf.show 域名使用 HTTPS。
    扫描成功后通过 --exec-cmd 直接执行指定命令并退出，避免进入交互模式。
    内部执行: python -m fenjing scan -u "<target_url>" --no-verify-ssl -e "<exec_cmd>"
    """
    url = (target_url or "").strip()
    if not url.startswith(("http://", "https://")):
        return "错误: target_url 必须以 http:// 或 https:// 开头"

    url = _normalize_url_for_fenjing(url)
    cmd = ["python", "-m", "fenjing", "scan", "-u", url, "--no-verify-ssl", "-e", exec_cmd]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return (
            "错误: 运行失败。请确认本机可执行 `python -m fenjing --help`，"
            "并确保已安装 FenJing（例如 `pipx install fenjing` 或在当前 Python 环境 pip 安装）。"
        )
    except subprocess.TimeoutExpired:
        return f"错误: FenJing 执行超时(>{timeout} 秒)，请减少任务复杂度或增大 timeout。"
    except Exception as e:
        return f"错误: 调用 FenJing 失败: {e}"

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    # 给 LLM 一个非常稳定的结构化输出
    out = [f"[cmd] {' '.join(cmd)}", f"[exit_code] {proc.returncode}"]
    if stdout:
        out.append("[stdout]\n" + stdout)
    if stderr:
        out.append("[stderr]\n" + stderr)
    return "\n\n".join(out)

