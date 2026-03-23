"""目录/文件扫描工具。"""
import hashlib
import os
import random
import string
from urllib.parse import urlparse

import requests

from langchain_core.tools import tool

from .constants import DIRSEARCH_MAX_FOUND, FORBIDDEN_CHARS

# 基线响应体取前 N 字节做指纹，用于识别「伪 200」（SPA 统一回传同一页面）
BASELINE_BODY_SAMPLE = 4096


@tool
def dirsearch_scan(target_url: str = "http://127.0.0.1"):
    """
    目录/文件爆破。target_url: 目标 URL。
    """
    # 1. 基本 URL 规范化与安全校验
    url = (target_url or "http://127.0.0.1").strip()
    if FORBIDDEN_CHARS.search(url):
        msg = "错误：target_url 含有非法字符"
        print(f"\n[dirsearch_scan] {msg}")
        return msg
    if not url.startswith(("http://", "https://")):
        msg = "错误：target_url 必须以 http:// 或 https:// 开头"
        print(f"\n[dirsearch_scan] {msg}")
        return msg

    # 2. 仅允许本地目标
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
    except Exception:
        msg = "错误：无法解析 target_url"
        print(f"\n[dirsearch_scan] {msg}")
        return msg
    if not host:
        msg = "错误：无法从 target_url 解析出 host"
        print(f"\n[dirsearch_scan] {msg}")
        return msg

    # 3. 从 dicc.txt 读取字典（优先 tools 同目录，否则项目根目录）
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.dirname(_this_dir)
    _dicc_path = os.path.join(_this_dir, "dicc.txt")
    if not os.path.isfile(_dicc_path):
        _dicc_path = os.path.join(_root_dir, "dicc.txt")
    if not os.path.isfile(_dicc_path):
        msg = f"错误：未找到字典文件 dicc.txt，请将 dicc.txt 放在项目根目录或 tools 目录下"
        print(f"\n[dirsearch_scan] {msg}")
        return msg
    try:
        with open(_dicc_path, "r", encoding="utf-8", errors="replace") as f:
            wordlist = [line.strip() for line in f if line.strip()]
    except Exception as e:
        msg = f"错误：读取 dicc.txt 失败：{e}"
        print(f"\n[dirsearch_scan] {msg}")
        return msg
    if not wordlist:
        msg = "错误：dicc.txt 为空或无可用的路径条目"
        print(f"\n[dirsearch_scan] {msg}")
        return msg

    # 4. 使用 Session 提升性能，并显式禁用系统代理，避免干扰本地请求
    session = requests.Session()
    session.trust_env = False

    # 5. 基线请求：很多站点（尤其 SPA）对任意路径都返回 200 + 同一页面，需用「响应指纹」过滤
    baseline_len = None
    baseline_fingerprint = None
    baseline_path = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=24)
    )
    try:
        base_resp = session.get(
            f"{url.rstrip('/')}/{baseline_path}",
            timeout=5,
            allow_redirects=False,
        )
        baseline_len = len(base_resp.content)
        sample = base_resp.content[:BASELINE_BODY_SAMPLE]
        baseline_fingerprint = hashlib.sha256(sample).hexdigest()
    except Exception:
        pass

    def _same_as_baseline(resp) -> bool:
        """与基线响应一致则视为「伪 200」，不当作有效命中。"""
        if baseline_len is None or baseline_fingerprint is None:
            return False
        if len(resp.content) != baseline_len:
            return False
        sample = resp.content[:BASELINE_BODY_SAMPLE]
        return hashlib.sha256(sample).hexdigest() == baseline_fingerprint

    found = []

    for path in wordlist:
        full_url = f"{url.rstrip('/')}/{path}"
        try:
            resp = session.get(full_url, timeout=5, allow_redirects=False)
        except Exception:
            continue

        code = resp.status_code
        if code in (301, 302, 403):
            found.append(f"[{code}] {full_url}")
            if len(found) >= DIRSEARCH_MAX_FOUND:
                break
        elif code == 200:
            # 200 需与基线比对，避免 SPA 对任意路径都返回同一页面造成的假阳性
            if not _same_as_baseline(resp):
                found.append(f"[{code}] {full_url}")
                if len(found) >= DIRSEARCH_MAX_FOUND:
                    break

    if not found:
        msg = "扫描完成，未发现常见敏感路径"
        # print(f"[dirsearch_scan] {msg}")
        return msg
    result = "\n".join(found)
    if len(found) >= DIRSEARCH_MAX_FOUND:
        result += f"\n(仅显示前 {DIRSEARCH_MAX_FOUND} 条，已截断)"
    print(f"\n[dirsearch_scan] 扫描结果:\n{result}")
    return result
