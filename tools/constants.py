"""工具包共用常量与配置。"""
import os
import re

FORBIDDEN_CHARS = re.compile(r'[;|&$`\'"<>\\\n\r\t]')
# dirsearch 单次扫描最多返回的命中条数，避免一次扫出过多导致上下文爆炸
DIRSEARCH_MAX_FOUND = 10
# 单次工具返回传入 LLM 的最大字符数（主流程截断用）
# 说明：若需要完整页面内容用于分析，可适当调大；过大可能导致上下文/日志膨胀。
MAX_TOOL_RESULT_CHARS = 50000
