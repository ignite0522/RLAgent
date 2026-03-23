"""CTF 联网辅助工具（2 个）：

1) ctf_search(query): 使用稀土掘金搜索 API 搜索关键词，返回最多 7 条 {title,url}
2) fetch_ctf_excerpt(urls, context): 在候选 URL 里挑最合适的一条，使用 Jina Reader 抓正文节选

说明：
- Jina Reader 支持可选 Authorization: Bearer <JINA_API_KEY>（从环境变量读取）
"""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import quote, quote_plus

import requests
from langchain_core.tools import tool


_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def _compact_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()


def _juejin_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """稀土掘金搜索 API：query -> {title,url,snippet} 列表。"""
    q = (query or "").strip()
    if not q:
        return []

    # 只保留回退路径：用 Jina Reader 抓掘金搜索页，再从文本里抽链接
    search_page = f"https://juejin.cn/search?query={quote_plus(q)}"
    jina_url = _jina_reader_url(search_page)
    if not jina_url:
        return []
    try:
        r = requests.get(jina_url, headers=_jina_headers(), timeout=25)
    except requests.RequestException:
        return []
    if r.status_code != 200 or not r.text:
        return []
    r.encoding = r.apparent_encoding or "utf-8"
    txt = r.text

    # 优先从 Markdown 链接中提取标题与 URL： [title](https://juejin.cn/post/xxxx)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    md_pat = re.compile(r"\[([^\]\n]{2,200})\]\((https?://juejin\.cn/post/[A-Za-z0-9]+)\)")
    for title_raw, url in md_pat.findall(txt):
        url = url.strip()
        if url in seen:
            continue
        seen.add(url)
        title = _compact_text(title_raw)
        out.append({"title": title, "url": url, "snippet": ""})
        if len(out) >= max_results:
            return out

    # 兜底：只抽 URL（标题留空）
    for m in re.finditer(r"https?://juejin\.cn/post/[A-Za-z0-9]+", txt):
        url = m.group(0).strip()
        if url in seen:
            continue
        seen.add(url)
        out.append({"title": "", "url": url, "snippet": ""})
        if len(out) >= max_results:
            break

    # 如果标题都为空，则用 Jina Reader 轻量补齐每条的 Title:（最多 max_results 条）
    title_pat = re.compile(r"(?im)^Title:\s*(.+?)\s*$")
    for item in out:
        if item.get("title"):
            continue
        u = (item.get("url") or "").strip()
        if not u:
            continue
        jina_u = _jina_reader_url(u)
        if not jina_u:
            continue
        try:
            rr = requests.get(jina_u, headers=_jina_headers(), timeout=15)
            if rr.status_code != 200 or not rr.text:
                continue
            mm = title_pat.search(rr.text)
            if mm:
                item["title"] = _compact_text(mm.group(1))
        except requests.RequestException:
            continue

    return out


@tool
def ctf_search(query: str, max_results: int = 7, max_chars: int = 2000) -> str:
    """
    通过稀土掘金搜索 API 搜索关键词，返回候选 URL 列表。

    参数：
    - query: 搜索关键词
    - max_results: 最大返回条数（强制 1~7，默认 7）
    - max_chars: 返回文本最大字符数，0 表示不限制

    返回：
    - `[ctf_search]` + JSON：`{"query":"...","results":[{"title","url"}...]}`
    """
    q = (query or "").strip()
    if not q:
        return "[ctf_search] query 不能为空"

    mr = int(max_results or 7)
    mr = max(1, min(mr, 7))
    hits = _juejin_search(q, mr)
    results = [{"title": (h.get("title") or ""), "url": (h.get("url") or "")} for h in hits]
    text = json.dumps({"query": q, "results": results}, ensure_ascii=False, indent=2)
    if max_chars and max_chars > 0:
        text = text[:max_chars]
    return f"[ctf_search]\n{text}"


def _jina_reader_url(url: str) -> str:
    """
    将任意 http(s) URL 包装成 Jina Reader URL：
    - https://example.com/a -> https://r.jina.ai/https://example.com/a
    """
    u = (url or "").strip()
    if u.startswith("http://") or u.startswith("https://"):
        # 保险起见做一次 URL 编码（保留 : / ? & = 等常见符号）
        return "https://r.jina.ai/" + quote(u, safe=":/?&=%#@+;,")
    return ""

def _jina_headers() -> dict:
    """
    可选：若设置了环境变量，则对 r.jina.ai 请求添加 Authorization 头。
    - JINA_API_KEY（推荐）
    - JINA_READER_API_KEY（兼容别名）
    """
    key = (os.getenv("JINA_API_KEY") or os.getenv("JINA_READER_API_KEY") or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _extract_readable_text(text: str) -> str:
    """
    尽可能把抓到的内容压成可读纯文本：
    - 若是 Jina 的 Markdown：优先提取 `Markdown Content:` 后的正文
    - 若是 HTML：去掉 script/style/标签
    - 若是纯文本：直接压缩空白
    """
    if not isinstance(text, str):
        return ""
    raw = text

    # 1. 若包含 Jina Reader 的 "Markdown Content:"，只保留正文部分
    m = re.search(r"Markdown Content:\s*(.*)", raw, re.S)
    if m:
        raw = m.group(1)

    # 2. 去掉常见 Markdown 图片噪音（![Image xx](...)）
    raw = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", raw)


    # 3. 压缩空白
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


@tool
def fetch_ctf_excerpt(
    urls: Any,
    context: str = "",
    excerpt_chars: int = 5000,
    max_chars: int = 4000,
) -> str:
    """
    在候选 URL 里挑选最适合当前上下文的一条，并抓取其正文节选。

    参数：
    - urls: 候选 URL 列表（Python list / 或 JSON 字符串均可）
    - context: 当前题目/环境描述（用于辅助挑选）
    - excerpt_chars: 正文节选最大长度（强制 <=5000，默认 5000；0 表示不抓正文）
    - max_chars: 返回文本最大字符数，0 表示不限制（默认 4000）

    返回：
    - `[fetch_ctf_excerpt]` + JSON：`{"chosen_url": "...", "excerpt": "...", "candidates": [...] }`
    """
    # Parse urls
    parsed: list[str] = []
    if isinstance(urls, list):
        parsed = [str(u).strip() for u in urls if str(u).strip()]
    elif isinstance(urls, str):
        s = urls.strip()
        if s.startswith("["):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    parsed = [str(u).strip() for u in arr if str(u).strip()]
            except Exception:
                parsed = []
        else:
            parsed = [s] if s else []
    else:
        parsed = [str(urls).strip()] if urls is not None else []

    parsed = [u for u in parsed if u.startswith("http://") or u.startswith("https://")]
    if not parsed:
        return "[fetch_ctf_excerpt] urls 为空或无有效 http(s):// URL"

    ec = int(excerpt_chars or 0)
    ec = max(0, min(ec, 5000))

    ctx = (context or "").lower()

    def score(u: str) -> int:
        s = 0
        lu = u.lower()
        if "juejin.cn/post/" in lu:
            s += 2
        if "ctfshow" in lu or "ctf.show" in lu:
            s += 3
        if "ssti" in lu:
            s += 2
        if "sql" in lu or "sqli" in lu:
            s += 2
        if "waf" in lu or "inject" in lu:
            s += 1
        if ctx:
            for kw in (
                "ssti",
                "jinja",
                "模板",
                "web",
                "ctfshow",
                "过滤",
                "绕过",
                "sql",
                "注入",
                "waf",
                "select-waf",
                "api/v5.php",
                "select id,username,password",
            ):
                if kw in ctx and kw in lu:
                    s += 1
        return s

    candidates = sorted(parsed, key=score, reverse=True)[:7]

    # 按候选顺序逐个尝试抓正文，优先选出“正文里包含上下文关键词”的文章
    ctx_tokens = []
    if ctx:
        for t in ("ctfshow", "ctf.show", "sql", "注入", "waf", "过滤", "绕过", "select-waf", "api/v5.php"):
            if t in ctx.lower() and t not in ctx_tokens:
                ctx_tokens.append(t)

    chosen_url = candidates[0]
    excerpt = ""
    best_match = -1
    for u in candidates:
        if ec <= 0:
            chosen_url = u
            excerpt = ""
            break
        jina_url = _jina_reader_url(u)
        if not jina_url:
            continue
        try:
            r = requests.get(jina_url, headers=_jina_headers(), timeout=25)
            if r.status_code != 200 or not r.text:
                continue
            r.encoding = r.apparent_encoding or "utf-8"
            tmp = _extract_readable_text(r.text)[:ec]
        except requests.RequestException:
            continue

        # 计算匹配度：命中上下文关键 token 越多越好，同时保证非空
        if not tmp:
            continue
        mcount = 0
        low = tmp.lower()
        for t in ctx_tokens:
            if t.lower() in low:
                mcount += 1
        if mcount > best_match:
            best_match = mcount
            chosen_url = u
            excerpt = tmp
            if best_match >= max(1, len(ctx_tokens) // 2):
                break

    payload = {"chosen_url": chosen_url, "excerpt": excerpt, "candidates": candidates}
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    if max_chars and max_chars > 0:
        text = text[:max_chars]

    return f"[fetch_ctf_excerpt]\n{text}"