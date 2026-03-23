"""根据漏洞类型读取项目内技术文档（防御与绕过），供 Agent 在识别到对应题型时调用。"""
from pathlib import Path

from langchain_core.tools import tool

# 项目根目录（tools 的上级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TECH_DOCS_DIR = _PROJECT_ROOT / "tech_docs"

# 漏洞类型 -> 文档文件名
_VULN_DOC_MAP = {
    "sql": "sql_injection.md",
    "sql_injection": "sql_injection.md",
    "xss": "xss.md",
    "file_include": "file_include.md",
    "lfi": "file_include.md",
    "rfi": "file_include.md",
    "php_deserialize": "php_deserialization.md",
    "php_deserialization": "php_deserialization.md",
    "php": "php_deserialization.md",
    "ssrf": "ssrf.md",
    "rce": "command_execution.md",
    "command_execution": "command_execution.md",
    "cmd": "command_execution.md",
    "命令执行": "command_execution.md",
}


@tool
def read_doc(vuln_type: str):
    """
    读取指定漏洞类型的技术文档（常见防御与绕过手段）。
    vuln_type: 漏洞类型，可选：sql / xss / php_deserialize / ssrf / rce（或 sql_injection、command_execution、命令执行 等别名）。
    当识别到题目属于某类漏洞时调用此工具，再根据文档内容进行测试。
    """
    key = vuln_type.strip().lower()
    filename = _VULN_DOC_MAP.get(key)
    if not filename:
        supported = "、".join(sorted(set(_VULN_DOC_MAP.keys())))
        return f"未找到类型「{vuln_type}」的文档。支持的类型：{supported}"
    path = _TECH_DOCS_DIR / filename
    if not path.is_file():
        return f"文档文件不存在: {path}"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"读取文档失败: {e}"