"""端口扫描工具。"""
import subprocess

from langchain_core.tools import tool


@tool
def nmap_scan(target: str, scan_type: str = "quick"):
    """
    端口扫描。target: 目标 IP/域名/URL；scan_type: quick(常用端口)/full(全端口)/service(服务探测)。
    """
    # 从 URL 中提取主机（如 http://example.com:8080/path -> example.com）
    host = target.strip()
    if "://" in host:
        host = host.split("://", 1)[1]
    if "/" in host:
        host = host.split("/", 1)[0]
    if ":" in host and not host[0].isdigit():
        host = host.rsplit(":", 1)[0]  # 保留端口前的域名

    scan_opts = {
        "quick": "-F -T4",           # 快速扫描常用端口
        "full": "-p- -T4",           # 全端口扫描
        "service": "-sV -T4 --top-ports 100",  # 服务版本探测
    }
    opts = scan_opts.get(scan_type, scan_opts["quick"])

    try:
        result = subprocess.run(
            ["nmap", *opts.split(), host],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0 and result.stderr:
            return f"nmap 执行失败: {result.stderr}"
        return result.stdout or "无输出"
    except FileNotFoundError:
        return "错误：未找到 nmap，请确保已安装 nmap 并加入 PATH"
    except subprocess.TimeoutExpired:
        return "错误：nmap 执行超时"
