import json
import os
import sys
import argparse
import re
from typing import TypedDict, Annotated, Any, Literal

import dotenv
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


from tools import (
    nmap_scan,
    dirsearch_scan,
    curl_request,
    php_run,
    python_run,
    sqlmap_scan,
    fenjing_ssti,
    check_xss,
    read_doc,
    ctf_search,
    fetch_ctf_excerpt,
    MAX_TOOL_RESULT_CHARS,
)
dotenv.load_dotenv()

# Agent 最多执行多少轮「LLM -> 工具 -> LLM」，防止无限循环
MAX_TOOL_ROUNDS = 444


# --- 步骤 1：图状态与节点 ---

class State(TypedDict, total=False):
    """图状态。
    - messages: 对话与工具调用消息
    - code_summary: 已读取页面/源码中与目标漏洞相关的摘要（关键函数、过滤/验证逻辑等），供后续每轮 LLM 优先参考
    - doc_content: read_doc 获取的技术文档内容（各漏洞的防御与绕过），供后续每轮 LLM 优先参考
    - search_notes: web_search_ctf 返回结果中的 excerpt 精炼汇总（最多前5000字），供后续每轮 LLM 参考
    - reports_board: 小D 每轮给大Q的“情况汇报”记录板（用于给大Q提供跨轮次记忆）
    - tool_rounds: 已执行的「LLM->工具->LLM」轮数，用于达到 MAX_TOOL_ROUNDS 后强制结束
    """

    messages: Annotated[list, add_messages]
    code_summary: str
    doc_content: str
    search_notes: str
    reports_board: list[str]
    tool_rounds: int


RUN_TOKEN_USAGE: dict[str, Any] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "by_stage": {},
}


def _extract_token_usage(ai_msg: Any) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    usage_meta = getattr(ai_msg, "usage_metadata", None) or {}
    if isinstance(usage_meta, dict):
        prompt_tokens = int(usage_meta.get("input_tokens", 0) or 0)
        completion_tokens = int(usage_meta.get("output_tokens", 0) or 0)
        total_tokens = int(usage_meta.get("total_tokens", 0) or 0)

    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        response_meta = getattr(ai_msg, "response_metadata", None) or {}
        token_usage = response_meta.get("token_usage", {}) if isinstance(response_meta, dict) else {}
        if isinstance(token_usage, dict):
            prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
            total_tokens = int(token_usage.get("total_tokens", 0) or 0)

    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": max(prompt_tokens, 0),
        "completion_tokens": max(completion_tokens, 0),
        "total_tokens": max(total_tokens, 0),
    }


def _accumulate_token_usage(stage: str, ai_msg: Any) -> None:
    usage = _extract_token_usage(ai_msg)
    RUN_TOKEN_USAGE["prompt_tokens"] += usage["prompt_tokens"]
    RUN_TOKEN_USAGE["completion_tokens"] += usage["completion_tokens"]
    RUN_TOKEN_USAGE["total_tokens"] += usage["total_tokens"]
    by_stage = RUN_TOKEN_USAGE.setdefault("by_stage", {})
    stage_usage = by_stage.setdefault(
        stage, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    stage_usage["prompt_tokens"] += usage["prompt_tokens"]
    stage_usage["completion_tokens"] += usage["completion_tokens"]
    stage_usage["total_tokens"] += usage["total_tokens"]


def _print_token_usage_summary() -> None:
    print("\n[token 使用统计]")
    print(
        f"总计: prompt={RUN_TOKEN_USAGE['prompt_tokens']}, "
        f"completion={RUN_TOKEN_USAGE['completion_tokens']}, "
        f"total={RUN_TOKEN_USAGE['total_tokens']}"
    )
    by_stage = RUN_TOKEN_USAGE.get("by_stage", {})
    if by_stage:
        print("分项:")
        for stage, usage in by_stage.items():
            print(
                f"- {stage}: prompt={usage.get('prompt_tokens', 0)}, "
                f"completion={usage.get('completion_tokens', 0)}, "
                f"total={usage.get('total_tokens', 0)}"
            )


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


def _strip_dsml_blocks(text: str) -> str:
    """移除模型输出中的 DSML/function_calls 伪标记，避免污染给大Q的汇报内容。"""
    if not text:
        return ""
    s = str(text)
    # 去掉形如：<｜DSML｜function_calls> ... </｜DSML｜function_calls>
    s = re.sub(r"<｜DSML｜function_calls>[\s\S]*?</｜DSML｜function_calls>", "", s)
    # 兜底：如果只出现零散 DSML 标签，也尽量剔除
    s = re.sub(r"<｜DSML｜[^>]*?>", "", s)
    s = re.sub(r"</｜DSML｜[^>]*?>", "", s)
    return s.strip()


# 通用 CTF Agent：支持信息收集、命令执行、SQL 注入、PHP 反序列化、SSRF、SSTI、XSS 等题型
tools = [
    nmap_scan,
    dirsearch_scan,
    curl_request,
    php_run,
    python_run,
    sqlmap_scan,
    fenjing_ssti,
    check_xss,
    read_doc,
    ctf_search,
    fetch_ctf_excerpt,
]
llm_worker = ChatOpenAI(
    model=os.getenv("REMOTE_WORKER_MODEL", "deepseek-chat"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    openai_api_base=os.getenv("REMOTE_WORKER_API_BASE", "https://api.deepseek.com"),
    request_timeout=120,
)
llm_with_tools = llm_worker.bind_tools(tools)

# 本地监督大模型（大Q），不再调用远端模型
# 默认走本机 OpenAI 兼容服务（例如 lora/http_server.py: http://127.0.0.1:8001/v1）
llm_supervisor = ChatOpenAI(
    model=os.getenv("LOCAL_SUPERVISOR_MODEL", "qwen-qctf-supervisor"),
    openai_api_key=os.getenv("LOCAL_LLM_API_KEY", "unused"),
    openai_api_base=os.getenv("LOCAL_LLM_API_BASE", "http://127.0.0.1:8001/v1"),
    request_timeout=120,
)


def _tool_catalog_text() -> str:
    """给大Q看的真实工具手册（名称/用途/关键参数）。"""
    lines: list[str] = [
        "可用工具仅限以下这些（不得编造其它工具名）。",
        "",
    ]
    for idx, t in enumerate(tools, start=1):
        name = getattr(t, "name", "") or ""
        desc = (getattr(t, "description", "") or "").strip()
        lines.append(f"【{idx}) {name}】")
        if desc:
            lines.append(desc)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _summarize_source(raw_content: str) -> str:
    """内部工具：对 read_web_file 读到的源码/页面内容做一次与漏洞利用相关的摘要。

    说明：
    - 不是对外暴露的 LangChain 工具，而是在工具执行后自动调用
    - 输出会存进 State 的 `code_summary` 字段中，供后续每轮对话使用
    """
    try:
        # 为避免超长页面源码把上下文撑爆，这里只取前一段给 LLM 做“压缩删减”
        # （具体截断长度可按需要再调）
        input_text = raw_content[:12000] if isinstance(raw_content, str) else str(raw_content)[:12000]

        system = SystemMessage(
            content=(
                "你是 网页源码压缩助手。"
                "只做“从网页源码中抽取可复用的关键线索”，不做漏洞结论推断；禁止编造。"
                "若无法从文本中抽取到可用线索，整段输出“未见关键线索”。"
                "请特别注意：输入文本可能以“文件内容:”或类似前缀开头，请忽略该前缀后再审读。"
            )
        )
        human = HumanMessage(
            content=(
                "下面是网页源码（可能非常长且包含大量无用字符）。请把它“压缩删减”成可复用的源码线索摘要。\n"
                "输出要求：仅输出一段中文、不要换行；总长度不超过180个汉字。\n"
                "严禁编造：只写你在输入文本里直接看到或能从结构明显确认的线索；若几乎没有可用线索输出“未见关键线索”。\n"
                "请优先按以下顺序覆盖（字数不够就只保留前几项）：\n"
                "1) 页面标题/title 或主要页面名（有就写）\n"
                "2) 输入入口：form 的 action/method、input/select/textarea 的 name/id（只写字段名）\n"
                "3) 前端交互线索：关键按钮/事件处理器（如 onclick/onchange）出现的函数名/事件名（只写关键字）\n"
                "4) 相关接口线索：href 与 script src（只写路径/片段）\n"
                "5) 若文本里直接出现危险相关词（如 eval/document.write/innerHTML/XMLHttpRequest/fetch 等），只提取“词/函数名”作为告警线索\n"
                f"\n=== 内容开始 ===\n{input_text}\n=== 内容结束 ==="
            )
        )
        resp = llm_supervisor.invoke([system, human])
        _accumulate_token_usage("summarize_source", resp)
        out = resp.content if hasattr(resp, "content") else str(resp)
        out = str(out).replace("\r", "").replace("\n", "").strip()
        if len(out) > 180:
            out = out[:180]
        return out or "未见关键线索"
    except Exception as e:
        return f"（源码摘要自动生成失败：{e}）"


def _summarize_tool_result(tool_name: str, raw_content: str) -> str:
    """对一次工具调用结果做精炼摘要，避免 ToolMessage 里塞入过长内容。"""
    try:
        system = SystemMessage(
            content=(
                "你是渗透测试日志压缩助手。"
                "请用不超过6行中文，总结下面这次工具调用的关键信息："
                "1）工具类型与大致目标/参数；2）关键发现（如状态码/报错/是否存在漏洞迹象/是否成功执行）；"
                "3）若是 HTML/源码，只描述与漏洞利用相关的要点，不要重复大段原文。"
                "若工具是 fenjing_ssti/sqlmap_scan，请额外给出：是否成功发现漏洞、可用参数/入口、可直接复用的关键 payload/命令、以及下一步建议。"
            )
        )
        human = HumanMessage(
            content=f"工具名: {tool_name}\n\n完整输出如下（可能很长，请只提炼要点）：\n\n{raw_content}"
        )
        resp = llm_supervisor.invoke([system, human])
        _accumulate_token_usage("summarize_tool_result", resp)
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        # 摘要失败时退回原始前 800 字，防止上下文爆炸
        prefix = raw_content[:800] if isinstance(raw_content, str) else str(raw_content)[:800]
        return f"（工具结果摘要自动生成失败：{e}，以下为原始前 800 字）\n{prefix}"


def chatbot(state: State, config: RunnableConfig | None = None) -> Any:
    """核心 LLM 节点。

    会把 `doc_content`（read_doc 结果）和 `code_summary` 放在系统提示最前面，让 LLM 每轮优先看到。
    """
    summary_prefix = ""
    search_notes = state.get("search_notes")
    if search_notes:
        summary_prefix += (
            "【以下是 web_search_ctf 搜索结果中的正文节选汇总（最多前5000字），用于快速获取思路：】\n"
            f"{search_notes}\n\n"
        )
    doc_content = state.get("doc_content")
    if doc_content:
        summary_prefix += (
            "【以下是 read_doc 获取的技术文档（防御与绕过），请优先按文档思路测试：】\n"
            f"{doc_content}\n\n"
        )
    code_summary = state.get("code_summary")
    if code_summary:
        summary_prefix += (
            "【以下是你此前读取到的与漏洞相关的页面/源码关键信息（输入点、关键函数、过滤/校验/编码逻辑等），请在后续利用时优先参考：】\n"
            f"{code_summary}\n\n"
        )

    try:
        base_dir = os.path.dirname(__file__)
        sys_prompt_path = os.path.join(base_dir, "tech_docs", "system_prompt.txt")
        with open(sys_prompt_path, "r", encoding="utf-8") as f:
            system_prompt_body = f.read()
    except Exception:
        system_prompt_body = (
            "你是一名 CTF 竞赛与渗透测试专家，负责综合分析并利用题目中的漏洞。"
        )




    tools_catalog = _tool_catalog_text()

    # --- 小D 先汇报，再由大Q决策 ---
    # 先由远端小模型小D阅读完整对话与摘要，整理成一份给大Q看的「情况汇报」，
    # 再让本地监督大模型大Q只基于这份汇报做决策，减少大Q的上下文负担。
    report_for_q = ""
    try:
        reporter_sys = SystemMessage(
            content=(
                summary_prefix
                + "【工具列表（重要）】\n"
                + tools_catalog
                + "\n"
                + "你现在扮演大模型小D。你要产出一份给上级大Q用作“状态输入(state)”的结构化汇报。\n"
                "\n"
                "【输出格式】不再强制 JSON。请用极短中文汇报（建议 200-400 字以内），按下面顺序组织即可：\n"
                "1）目标：一句话\n"
                "2）发现：1-3 条要点\n"
                "3）下一步：明确写“工具名 + 关键参数草案 + 目的”\n"
                "4）给大Q的问题：没有就写“无”\n"
                "\n"
                "【硬性约束】\n"
                "- 工具名必须严格来自工具列表；不确定就填空字符串。\n"
            )
        )
        reporter_messages = [reporter_sys] + state["messages"]
        report_reply = llm_worker.invoke(reporter_messages)
        _accumulate_token_usage("reporter_small_d", report_reply)
        report_for_q = _strip_dsml_blocks(getattr(report_reply, "content", "") or "")
        try:
            GREEN = "\033[32m"
            RESET = "\033[0m"
            compact_report = " ".join(
                line.strip() for line in str(report_for_q).splitlines() if line.strip()
            )
            print(f"\n{GREEN}[小D汇报]:{RESET} {compact_report}")
        except Exception:
            pass
    except Exception as e:
        # 汇报失败：直接抛错中断，避免静默降级导致大Q决策依据不可靠
        raise RuntimeError(f"生成小D→大Q汇报失败：{e}") from e

    # --- 记录板：把每轮小D汇报写入 state.reports_board，供后续大Q跨轮次参考 ---
    board = list(state.get("reports_board") or [])
    if report_for_q:
        # 压成单段，减少上下文占用
        compact = " ".join(line.strip() for line in str(report_for_q).splitlines() if line.strip())
        if compact:
            board.append(compact)
    # 控制长度：只保留最近 12 条；同时限制总字符数，避免无限增长
    board = board[-12:]
    max_total_chars = 12_000
    while board and sum(len(x) for x in board) > max_total_chars:
        board.pop(0)

    # --- 大Q（监督大模型） ---
    # 大Q不再直接看所有对话，只看小D整理好的汇报与必要的结构化指令格式说明，
    # 然后给出下一步作战计划与工具选择。
    supervisor_sys = SystemMessage(
        content=(
            "你是本地大模型大Q，负责给远端小模型小D下达下一步“动作选择”（只选工具，不写长篇建议）。\n"
            "你的目标：只基于小D的「情况汇报」与工具列表，选择下一步最合适的一个工具。\n"
            "你有一个“记录板”，里面包含小D历轮汇报（最近若干条）。请先看记录板把上下文串起来，再决定本轮动作。\n"
            "\n"
            "你需要：\n"
            "1）认真阅读下面小D整理的「情况汇报」；\n"
            "2）从工具列表中选择**一个**工具名，作为下一步动作。\n"
            "\n"
            "【输出格式｜只能输出 JSON】\n"
            "{\"must_use_tool\": \"<工具名或空字符串>\"}\n"
            "- must_use_tool 为空字符串时，表示本轮不强制调用工具（例如需要小D先做纯分析/总结）。\n"
            "\n"
            "【重要约束】must_use_tool 必须严格从“工具列表”里选择一个名字；如果不确定，就返回空字符串。\n"
            "下面是工具列表与小D刚刚整理好的情况汇报，请基于这些信息来做判断，不要要求查看原始日志。"
        )
    )
    supervisor_messages = [
        supervisor_sys,
        HumanMessage(
            content=(
                "【工具列表（名称:用途）】\n"
                f"{tools_catalog}\n\n"
                "【记录板｜小D历轮汇报（由旧到新，可能已截断）】\n"
                + ("\n".join(f"- {x}" for x in board) if board else "（空）")
                + "\n\n"
                "【小D 情况汇报】\n"
                f"{report_for_q}"
            )
        ),
    ]
    supervisor_reply = llm_supervisor.invoke(supervisor_messages)
    _accumulate_token_usage("supervisor_big_q", supervisor_reply)
    supervisor_content = getattr(supervisor_reply, "content", "") or ""

    # 默认不强制工具，由小D自行判断；如果大Q给出结构化 must_use_tool，则尝试解析并强制绑定
    forced_tool_name = ""
    if supervisor_content:
        try:
            # 尝试从纯 JSON 或前缀文本中提取 JSON
            text = str(supervisor_content).strip()
            json_start = text.find("{")
            json_end = text.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                obj = json.loads(text[json_start : json_end + 1])
                forced_tool_name = (obj.get("must_use_tool") or "").strip()
        except Exception:
            forced_tool_name = ""
    if forced_tool_name:
        tools_by_name = {tool.name: tool for tool in tools}
        if forced_tool_name not in tools_by_name:
            forced_tool_name = ""

    # --- 小D（远端工具执行模型）根据大Q指令行动 ---
    # 小D的系统提示里显式嵌入大Q的最新决策，并用语气强调“必须服从大Q”。
    worker_sys = SystemMessage(
        content=(
            summary_prefix
            + system_prompt_body
            + "【工具列表（重要）】\n"
            + tools_catalog
            + "\n"
            + "【本地大模型大Q最新动作】大Q只输出一个 JSON 动作：{\"must_use_tool\": \"<工具名或空字符串>\"}。\n"
            + "你的最高优先级是：先解析并承接该 JSON 中的 must_use_tool，然后**优先按该工具行动**。\n"
            + "若你判断该动作在当前环境下会导致明显无效/浪费/风险（例如工具不可用、参数缺失无法补全、与目标阶段完全不匹配），才允许反对，但必须按下面格式说明理由与替代动作。\n"
            + "大Q 的动作如下（原文照抄，便于你解析）：\n"
              f"{supervisor_content}\n\n"
            + "\n\n你现在是远端小模型小D，职责是：在保证安全与合规的前提下，尽量高效推进渗透测试，并尊重大Q的指导。"
            "如你与大Q观点不同，可以提出，但必须给出清晰理由与替代方案。"
            "\n\n【协议策略】默认优先尝试 HTTPS；若 HTTPS 因证书/握手/连接失败导致无法获取有效响应，请立即改用 HTTP 继续验证与测试，并在回复中说明原因。"
            "\n\n【对话格式｜必须先承接大Q】\n"
            "你每一轮回复开头必须先用 1-2 句承接大Q：\n"
            "1）用“收到大Q动作：must_use_tool=... ”明确写出你解析到的工具名（或空字符串）；\n"
            "2）说明你将如何执行。\n"
            "如果你决定不按大Q建议执行，必须紧接着写【反对理由】与【替代方案】，然后再行动。\n"
            "\n"
            "【硬性约束】\n"
            "- 你只能调用工具列表中的工具；不得编造工具名或自行假设存在 `post_http/whatweb_scan/gobuster` 等工具。\n"
            "- 若要读取页面/源码用于审计摘要：使用 `curl_request` 并设置 `as_source=true`。\n"
            "- 若要发送请求/带 Cookie/进行文件上传：使用 `curl_request`；可以选择提供 `curl_args`，也可以用结构化参数（method/data/json_data/upload_* 等）。\n"
            "- `curl_request` 的返回模式由 `as_source` 控制：as_source=false 用于正常 HTTP 请求结果摘要。\n"
            "\n\n【重要规则｜允许反对但必须解释】\n"
            "如果大Q在 JSON 中给出了 must_use_tool 且该工具在工具列表内，但你认为此时不应执行该工具、需要改用其它工具：\n"
            "你可以反对，但必须在本轮回复里先用中文写清楚两段：\n"
            "1）【反对理由】为什么此刻执行 must_use_tool 会导致风险/无效/浪费；\n"
            "2）【替代方案】你将改用哪个工具（必须是工具列表中的名字）以及你打算怎么调用。\n"
            "写完这两段后，再发起你的 tool_calls。不要不解释就换工具。"
            "\n\n【重要规则｜尽可能先说话再调用工具】\n"
            "只要你准备发起任何 tool_calls，请尽量先输出 1-3 句中文说明：\n"
            "1）你要调用哪个工具（或哪些工具）；\n"
            "2）你为什么要这么做（目的/假设）；\n"
            "3）你预期从结果里确认什么、下一步怎么接。\n"
            "（如果模型协议导致 content 为空，也尽量在下一轮补充说明。）"
        )
    )
    worker_messages = [worker_sys] + state["messages"]

    # 不再强制 tool_choice：始终全工具模式，让小D自然输出文字并自行选择工具
    ai_message = llm_with_tools.invoke(worker_messages)
    _accumulate_token_usage("worker_small_d", ai_message)

    # 实时在终端打印 AI 的想法和计划调用的工具（绿色），便于调试与观察推理过程
    try:
        content_str = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
        BLUE = "\033[34m"
        PURPLE = "\033[35m"
        RESET = "\033[0m"

        # 打印大Q与小D的关键对话片段
        print(f"\n{BLUE}[大Q]:{RESET}")
        try:
            compact_q = " ".join(
                line.strip() for line in str(supervisor_content).splitlines() if line.strip()
            )
        except Exception:
            compact_q = str(supervisor_content)
        print(compact_q)
        print(f"{BLUE}[小D]:{RESET}")
        # 仅用于终端展示：把多行内容压成一段，避免每句都单独换行
        try:
            compact = " ".join(line.strip() for line in str(content_str).splitlines() if line.strip())
        except Exception:
            compact = str(content_str)
        if not compact and getattr(ai_message, "tool_calls", None):
            compact = "（本轮小D未输出文本，仅发起了工具调用）"
        print(compact)
        if getattr(ai_message, "tool_calls", None):
            print(f"{PURPLE}[tool_calls]:{RESET}")
            for tc in ai_message.tool_calls:
                print(f"- {tc.get('name')} args={tc.get('args')}\n\n")
    except Exception:
        # 打印日志失败时不要影响主流程
        pass

    # search_notes 只给“下一轮”看：本轮用完后立刻清空，确保再下一轮看不到
    return {"messages": [ai_message], "search_notes": "", "reports_board": board}


def final_llm(state: State, config: RunnableConfig | None = None) -> Any:
    """达到最大工具轮数后强制要求给出结论，不再调用工具。"""
    summary_prefix = ""
    if state.get("doc_content"):
        summary_prefix += "【技术文档】\n" + state["doc_content"][:2000] + "\n\n"
    if state.get("code_summary"):
        summary_prefix += "【源码摘要】\n" + state["code_summary"][:1500] + "\n\n"
    sys_msg = SystemMessage(
        content=summary_prefix
        + f"已达最大工具轮数（{MAX_TOOL_ROUNDS}），请**直接给出最终结论**，不要调用任何工具。简要总结已尝试的输入点、payload 与 check_xss 结果，并给出是否发现 XSS 的结论。"
    )

    # 为避免 OpenAI 对话协议错误（末尾 assistant 带 tool_calls 但没有对应 ToolMessage），
    # 需要去掉结尾那些尚未执行工具的 AI tool_calls 消息。
    history = list(state["messages"])
    while (
        history
        and getattr(history[-1], "type", "") == "ai"
        and getattr(history[-1], "tool_calls", None)
    ):
        history.pop()

    messages = [sys_msg] + history
    ai_message = llm_supervisor.invoke(messages)
    _accumulate_token_usage("final_llm", ai_message)
    return {"messages": [ai_message]}



def tool_executor(state: State, config: RunnableConfig | None = None) -> Any:
    tools_by_name = {tool.name: tool for tool in tools}
    ai_message = state["messages"][-1]
    tool_calls = ai_message.tool_calls

    messages = []
    accumulated_summary = state.get("code_summary", "")
    accumulated_doc = state.get("doc_content", "")
    next_search_notes = ""

    for tool_call in tool_calls:
        # 正常执行远端发起的所有 tool_calls（在强制绑定模式下通常只有 1 个）
        tool = tools_by_name[tool_call["name"]]
        raw_content = tool.invoke(tool_call["args"])

        # 为对话历史准备的内容：
        # - 大多数工具：直接保留原始文本，只在超过 MAX_TOOL_RESULT_CHARS 时做截断
        # - sqlmap_scan/fenjing_ssti：输出可能极长，超过上限时先用 LLM 做摘要，再保留摘要
        display_content = raw_content
        if isinstance(raw_content, str) and len(raw_content) > MAX_TOOL_RESULT_CHARS:
            if tool_call["name"] in ("sqlmap_scan", "fenjing_ssti"):
                display_content = _summarize_tool_result(tool_call["name"], raw_content)
                if len(display_content) > MAX_TOOL_RESULT_CHARS:
                    display_content = display_content[:MAX_TOOL_RESULT_CHARS] + "\n\n...(结果已截断，超出上下文限制)"
            else:
                display_content = raw_content[:MAX_TOOL_RESULT_CHARS] + "\n\n...(结果已截断，超出上下文限制)"

        # 联网搜索/抓正文：终端直接打印结构化结果，便于人眼查看
        if tool_call["name"] in ("ctf_search", "fetch_ctf_excerpt") and isinstance(display_content, str):
            print(f"\n[{tool_call['name']} 结果]")
            print(display_content)

        # 把 fetch_ctf_excerpt(chosen_url, excerpt) 写入 state.search_notes（只给下一轮看）
        if tool_call["name"] == "fetch_ctf_excerpt" and isinstance(display_content, str):
            try:
                payload = display_content
                if payload.startswith("[fetch_ctf_excerpt]"):
                    payload = payload.split("\n", 1)[1] if "\n" in payload else ""
                data = json.loads(payload) if payload.strip().startswith("{") else {}
                chosen_url_val = (data.get("chosen_url") or "").strip()
                excerpt = (data.get("excerpt") or "").strip()
                if excerpt:
                    excerpt = excerpt[:5000]
                    block = f"来源: {chosen_url_val}\n{excerpt}".strip()
                    next_search_notes = (next_search_notes + "\n\n---\n\n" + block).strip() if next_search_notes else block
                    next_search_notes = next_search_notes[:10000]
            except Exception:
                pass

        # 用原始结果更新内部摘要记忆（不会进对话历史，避免上下文爆炸）
        if (
            tool_call["name"] == "curl_request"
            and tool_call.get("args", {}).get("as_source")
            and isinstance(raw_content, str)
            and raw_content.strip()
        ):
            summary = _summarize_source(raw_content)
            accumulated_summary = accumulated_summary + "\n\n---\n\n" + summary if accumulated_summary else summary
        if tool_call["name"] == "read_doc" and isinstance(raw_content, str) and raw_content.strip():
            accumulated_doc = accumulated_doc + "\n\n---\n\n" + raw_content if accumulated_doc else raw_content
        messages.append(
            ToolMessage(
                tool_call_id=tool_call["id"],
                content=json.dumps(display_content, ensure_ascii=False) if not isinstance(display_content, str) else display_content,
                name=tool_call["name"],
            )
        )

    result: State = {
        "messages": messages,
        "tool_rounds": state.get("tool_rounds", 0) + 1,
    }
    if accumulated_summary:
        result["code_summary"] = accumulated_summary
    if accumulated_doc:
        result["doc_content"] = accumulated_doc
    if next_search_notes:
        result["search_notes"] = next_search_notes
    return result




def route(state: State, config: RunnableConfig | None = None) -> Literal["tool_executor", "final_llm", "__end__"]:
    ai_message = state["messages"][-1]
    tool_rounds = state.get("tool_rounds", 0)
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        if tool_rounds >= MAX_TOOL_ROUNDS:
            return "final_llm"
        return "tool_executor"
    return END



def print_banner_colored() -> None:
    art = [
    " ██████╗████████╗███████╗ █████╗  ██████╗ ███████╗███╗   ██╗████████╗",
    "██╔════╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝",
    "██║        ██║   █████╗  ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║",
    "██║        ██║   ██╔══╝  ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║",
    "╚██████╗   ██║   ██║     ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║",
    " ╚═════╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝",
    "=========================Author:CUIT-矛盾实验室=========================="
    ]
    DEEP_GREEN = "\033[38;5;22m"
    RESET = "\033[0m"

    banner = "\n".join(art)
    if sys.stdout.isatty():
        print(DEEP_GREEN + banner + RESET, end="")
    else:
        print(banner, end="")


# --- 步骤 3：构建与运行 ---

def main() -> None:
    global RUN_TOKEN_USAGE
    RUN_TOKEN_USAGE = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_stage": {},
    }

    parser = argparse.ArgumentParser(description="通用 CTF 渗透测试 Agent")
    parser.add_argument(
        "-t",
        "--target",
        required=True,
        help="目标地址，例如 http://xxx.challenge.ctf.show/ 或 http://127.0.0.1:8080/",
    )
    args = parser.parse_args()

    graph_builder = StateGraph(State)
    graph_builder.add_node("llm", chatbot)
    graph_builder.add_node("tool_executor", tool_executor)
    graph_builder.add_node("final_llm", final_llm)


    graph_builder.add_edge(START, "llm")
    graph_builder.add_edge("tool_executor", "llm")
    graph_builder.add_conditional_edges("llm", route)
    graph_builder.add_edge("final_llm", END)

    graph = graph_builder.compile()
    print_banner_colored()

    target = args.target
    state = graph.invoke(
        {
            "messages": [
                (
                    "human",
                    f"目标为 {target}。"
                    f"尽量拿到 flag 或给出是否存在关键漏洞的最终结论。",
                )
            ]
        },
        config={"recursion_limit": max(200, MAX_TOOL_ROUNDS * 2 + 20)},
    )
    _print_token_usage_summary()



if __name__ == "__main__":
    main()