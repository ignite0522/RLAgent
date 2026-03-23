import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from tools import (
    MAX_TOOL_RESULT_CHARS,
    check_xss,
    ctf_search,
    curl_request,
    dirsearch_scan,
    fenjing_ssti,
    fetch_ctf_excerpt,
    nmap_scan,
    php_run,
    python_run,
    read_doc,
    sqlmap_scan,
)
from runtime_skills_loader import build_skills_block

dotenv.load_dotenv()

MAX_TOOL_ROUNDS = 444
USE_SUPERVISOR = True

LLM_RETRY_MAX_ATTEMPTS = int(os.getenv("LLM_RETRY_MAX_ATTEMPTS", "4"))
LLM_RETRY_BASE_DELAY_SECONDS = float(os.getenv("LLM_RETRY_BASE_DELAY_SECONDS", "1.0"))
LLM_RETRY_MAX_DELAY_SECONDS = float(os.getenv("LLM_RETRY_MAX_DELAY_SECONDS", "8.0"))


class State(TypedDict, total=False):
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


def _is_retryable_llm_error(err: Exception) -> bool:
    status_code = getattr(err, "status_code", None)
    if isinstance(status_code, int) and status_code >= 500:
        return True
    text = f"{type(err).__name__}: {err}".lower()
    keywords = (
        " 502",
        " 503",
        " 504",
        "internalservererror",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection reset",
        "connection error",
        "service unavailable",
        "bad gateway",
    )
    return any(k in text for k in keywords)


def _invoke_with_retry(model: Any, messages: list[Any], stage: str) -> Any:
    max_attempts = max(1, LLM_RETRY_MAX_ATTEMPTS)
    for attempt in range(1, max_attempts + 1):
        try:
            return model.invoke(messages)
        except Exception as e:
            can_retry = _is_retryable_llm_error(e) and attempt < max_attempts
            if not can_retry:
                raise
            delay = min(
                LLM_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
                LLM_RETRY_MAX_DELAY_SECONDS,
            )
            print(
                f"[LLM重试] stage={stage}, attempt={attempt}/{max_attempts}, "
                f"原因={type(e).__name__}: {e}, {delay:.1f}s后重试..."
            )
            time.sleep(delay)
    raise RuntimeError(f"{stage} 调用失败")


def _strip_dsml_blocks(text: str) -> str:
    if not text:
        return ""
    s = str(text)
    s = re.sub(r"<｜DSML｜function_calls>[\s\S]*?</｜DSML｜function_calls>", "", s)
    s = re.sub(r"<｜DSML｜[^>]*?>", "", s)
    s = re.sub(r"</｜DSML｜[^>]*?>", "", s)
    return s.strip()


def _wrap_poml(*, role: str, task: str, output_format: str) -> str:
    role = (role or "").strip()
    task = (task or "").strip()
    output_format = (output_format or "").strip()
    return "\n".join(
        [
            "<poml>",
            f"<role>{role}</role>" if role else "",
            f"<task>{task}</task>" if task else "",
            f"<output-format>{output_format}</output-format>" if output_format else "",
            "</poml>",
        ]
    ).strip()


def _extract_tool_and_reason_from_json(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    try:
        s = str(text).strip()
        l = s.find("{")
        r = s.rfind("}")
        if l == -1 or r <= l:
            return "", ""
        obj = json.loads(s[l : r + 1])
        tool = str(obj.get("must_use_tool") or "").strip()
        reason = str(obj.get("reason") or "").strip()
        return tool, reason
    except Exception:
        return "", ""


def _print_agent_turn(ai_message: Any, supervisor_content: str = "") -> None:
    """终端打印大Q/小D对话与工具调用，便于调试观察。"""
    try:
        content_str = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
        BLUE = "\033[34m"
        PURPLE = "\033[35m"
        RESET = "\033[0m"
        print(f"\n{BLUE}[大Q]:{RESET}")
        try:
            compact_q = " ".join(line.strip() for line in str(supervisor_content).splitlines() if line.strip())
        except Exception:
            compact_q = str(supervisor_content)
        print(compact_q if compact_q else "（本轮未启用大Q或无动作）")
        print(f"{BLUE}[小D]:{RESET}")
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
        pass


def _to_supervisor_safe_messages(messages: list[Any]) -> list[Any]:
    """适配本地 http_server: 仅保留 system/user/assistant 角色。"""
    safe: list[Any] = []
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        if msg_type == "system":
            safe.append(SystemMessage(content=str(getattr(msg, "content", "") or "")))
        elif msg_type in ("human", "user"):
            safe.append(HumanMessage(content=str(getattr(msg, "content", "") or "")))
        elif msg_type == "tool":
            tool_name = getattr(msg, "name", "") or ""
            tool_content = str(getattr(msg, "content", "") or "")
            safe.append(HumanMessage(content=f"[工具输出:{tool_name}]\n{tool_content}"))
        else:
            safe.append(HumanMessage(content=str(getattr(msg, "content", "") or "")))
    return safe


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

llm_supervisor = ChatOpenAI(
    model=os.getenv("LOCAL_SUPERVISOR_MODEL", "qwen-qctf-supervisor"),
    openai_api_key=os.getenv("LOCAL_LLM_API_KEY", "unused"),
    openai_api_base=os.getenv("LOCAL_LLM_API_BASE", "http://127.0.0.1:8001/v1"),
    request_timeout=120,
)


def _get_aux_llm() -> Any:
    """辅助摘要模型：有大Q时用本地大Q，无大Q时回退小D。"""
    return llm_supervisor if USE_SUPERVISOR else llm_worker


def _tool_catalog_text() -> str:
    lines = ["可用工具仅限以下这些（不得编造其它工具名）。", ""]
    for idx, t in enumerate(tools, start=1):
        name = getattr(t, "name", "") or ""
        desc = (getattr(t, "description", "") or "").strip()
        lines.append(f"【{idx}) {name}】")
        if desc:
            lines.append(desc)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _summarize_source(raw_content: str) -> str:
    try:
        input_text = raw_content[:12000] if isinstance(raw_content, str) else str(raw_content)[:12000]
        resp = _invoke_with_retry(
            _get_aux_llm(),
            [
                SystemMessage(content="你是网页源码压缩助手。只提取关键线索，若没有则输出“未见关键线索”。"),
                HumanMessage(content=f"请将以下源码压缩成不超过180字中文摘要：\n{input_text}"),
            ],
            "summarize_source",
        )
        _accumulate_token_usage("summarize_source", resp)
        out = str(getattr(resp, "content", "") or "").replace("\n", "").strip()
        return out[:180] if out else "未见关键线索"
    except Exception as e:
        return f"（源码摘要自动生成失败：{e}）"


def _summarize_tool_result(tool_name: str, raw_content: str) -> str:
    try:
        resp = _invoke_with_retry(
            _get_aux_llm(),
            [
                SystemMessage(content="你是渗透日志压缩助手，请在6行内输出关键结论。"),
                HumanMessage(content=f"工具名: {tool_name}\n\n原始输出:\n{raw_content}"),
            ],
            "summarize_tool_result",
        )
        _accumulate_token_usage("summarize_tool_result", resp)
        return str(getattr(resp, "content", "") or "")
    except Exception as e:
        return f"（工具结果摘要自动生成失败：{e}）\n{str(raw_content)[:800]}"


def chatbot(state: State, config: RunnableConfig | None = None) -> Any:
    summary_prefix = ""
    if state.get("search_notes"):
        summary_prefix += f"【搜索摘录】\n{state['search_notes']}\n\n"
    if state.get("doc_content"):
        summary_prefix += f"【技术文档】\n{state['doc_content']}\n\n"
    if state.get("code_summary"):
        summary_prefix += f"【源码摘要】\n{state['code_summary']}\n\n"

    try:
        with open(os.path.join(os.path.dirname(__file__), "tech_docs", "system_prompt.txt"), "r", encoding="utf-8") as f:
            system_prompt_body = f.read()
    except Exception:
        system_prompt_body = "你是一名 CTF 渗透测试专家。"

    tools_catalog = _tool_catalog_text()
    context_text = (
        (state.get("search_notes") or "")
        + "\n\n"
        + (state.get("doc_content") or "")
        + "\n\n"
        + (state.get("code_summary") or "")
    ).strip()
    skills_block = build_skills_block(base_dir=Path(__file__).resolve().parent, context_text=context_text)

    # 关闭大Q：直接小D决策并调用工具
    if not USE_SUPERVISOR:
        worker_sys = SystemMessage(
            content=_wrap_poml(
                role="你现在是远端小模型小D。",
                task=(
                    summary_prefix
                    + system_prompt_body
                    + "\n\n【工具列表（重要）】\n"
                    + tools_catalog
                    + "\n"
                    + "你现在是远端小模型小D，职责是：在保证安全与合规的前提下，尽量高效推进渗透测试。"
                    + "\n"
                    + "【协议策略】默认优先尝试 HTTPS；若 HTTPS 因证书/握手/连接失败导致无法获取有效响应，请立即改用 HTTP 继续验证与测试，并在回复中说明原因。\n"
                    + "【硬性约束】\n"
                    + "- 你只能调用工具列表中的工具；不得编造工具名或自行假设存在 `post_http/whatweb_scan/gobuster` 等工具。\n"
                    + "- 若要读取页面/源码用于审计摘要：使用 `curl_request` 并设置 `as_source=true`。\n"
                    + "- 若要发送请求/带 Cookie/进行文件上传：使用 `curl_request`；由你决定是否提供 `curl_args` 或使用结构化参数（method/data/json_data/upload_* 等）。\n"
                    + "- `curl_request` 返回模式由 `as_source` 控制：as_source=false 时用于正常 HTTP 请求结果摘要。\n"
                    + ("\n\n【运行时 Skills（自动注入）】\n" + skills_block if skills_block else "")
                ),
                output_format="只要准备发起 tool_calls，请先用 1-3 句中文说明：要调用哪个工具、目的是什么、预期确认什么。",
            )
        )
        worker_messages = [worker_sys] + state["messages"]
        ai_message = _invoke_with_retry(llm_with_tools, worker_messages, "worker_only_mode")
        _accumulate_token_usage("worker_only_mode", ai_message)
        _print_agent_turn(ai_message, "")
        return {"messages": [ai_message], "search_notes": ""}

    # 开启大Q：小D汇报 -> 大Q选工具 -> 小D执行
    report_reply = _invoke_with_retry(
        llm_worker,
        [SystemMessage(content=summary_prefix + "请输出给大Q的简短汇报。")] + state["messages"],
        "reporter_small_d",
    )
    _accumulate_token_usage("reporter_small_d", report_reply)
    report_for_q = _strip_dsml_blocks(getattr(report_reply, "content", "") or "")

    board = list(state.get("reports_board") or [])
    if report_for_q:
        board.append(" ".join(x.strip() for x in str(report_for_q).splitlines() if x.strip()))
    board = board[-12:]

    supervisor_messages = [
        SystemMessage(
            content=_wrap_poml(
                role="你是本地监督模型大Q。",
                task=(
                    "你是本地大模型大Q，负责参考远端小模型小D的汇报，然后给他下达下一步“动作选择”（只选工具，不写长篇建议）。\n"
                    "你是一名资深网络安全与CTF专家，熟悉常见 Web 漏洞（如 SQL 注入、XSS、SSRF、命令执行等）的利用与测试流程。\n"
                    "在选择工具时，要综合效率与价值：优先选择能验证关键假设或推进获取 flag 的动作，避免明显重复或低收益的探测。"
                ),
                output_format=(
                    "【输出格式｜只能输出 JSON】\n"
                    "{\"must_use_tool\": \"<工具名或空字符串>\", \"reason\": \"<必须使用时只给一句原因；must_use_tool为空时可为空或只给一句简短原因>\"}\n"
                    "- must_use_tool 为空字符串时，表示本轮不强制调用工具（例如需要小D先做纯分析/总结）。\n"
                    "【重要约束】must_use_tool 必须严格从“工具列表”里选择一个名字；如果不确定，就返回空字符串。\n"
                    "大Q的输出内容只包含：must_use_tool 与 reason，不输出其它文本。"

                ),
            )
        ),
        HumanMessage(
            content=(
                f"【工具列表】\n{tools_catalog}\n\n"
                f"【记录板】\n{chr(10).join('- ' + x for x in board) if board else '（空）'}\n\n"
                f"【本轮汇报】\n{report_for_q}"
            )
        ),
    ]
    supervisor_messages = _to_supervisor_safe_messages(supervisor_messages)
    supervisor_reply = _invoke_with_retry(llm_supervisor, supervisor_messages, "supervisor_big_q")
    _accumulate_token_usage("supervisor_big_q", supervisor_reply)
    supervisor_content_raw = str(getattr(supervisor_reply, "content", "") or "")
    must_use_tool, reason = _extract_tool_and_reason_from_json(supervisor_content_raw)
    valid_tools = {getattr(t, "name", "") for t in tools}
    if must_use_tool and must_use_tool not in valid_tools:
        must_use_tool = ""
        reason = "工具名不在可用工具列表，已回退为空。"
    supervisor_content = json.dumps(
        {"must_use_tool": must_use_tool, "reason": reason},
        ensure_ascii=False,
    )

    worker_sys = SystemMessage(
        content=_wrap_poml(
            role="你现在是远端小模型小D。",
            task=(
                summary_prefix
                + system_prompt_body
                + "\n\n【工具列表（重要）】\n"
                + tools_catalog
                + "\n"
                + "【本地大模型大Q最新动作】大Q输出一个 JSON 动作，至少包含：{\"must_use_tool\": \"<工具名或空字符串>\"}（可选包含 reason）。\n"
                + "你的最高优先级是：先解析并承接该 JSON 中的 must_use_tool，然后优先按该工具行动。\n"
                + "若你判断该动作在当前环境下会导致明显无效/浪费/风险（例如工具不可用、参数缺失无法补全、与目标阶段完全不匹配），才允许反对，但必须按下面格式说明理由与替代动作。\n"
                + "大Q 的动作和原因如下（原文照抄，便于你解析）：\n"
                + supervisor_content
                + "\n\n"
                + "你现在是远端小模型小D，职责是：在保证安全与合规的前提下，尽量高效推进渗透测试，并尊重大Q的指导。"
                + "如你与大Q观点不同，可以提出，但必须给出清晰理由与替代方案。\n"
                + "\n"
                + "【协议策略】默认优先尝试 HTTPS；若 HTTPS 因证书/握手/连接失败导致无法获取有效响应，请立即改用 HTTP 继续验证与测试，并在回复中说明原因。\n"
                + "\n"
                + "【对话格式｜必须先承接大Q】\n"
                + "你每一轮回复开头必须先用 1-2 句承接大Q：\n"
                + "1）用“收到大Q动作：must_use_tool=... ”明确写出你解析到的工具名（或空字符串）；\n"
                + "2）说明你将如何执行以及接下来选择什么工具继续利用。\n"
                + "如果你决定不按大Q建议执行，必须紧接着写【反对理由】与【替代方案】，然后再行动。\n"
                + "\n"
                + "【硬性约束】\n"
                + "- 你只能调用工具列表中的工具；不得编造工具名或自行假设存在 `post_http/whatweb_scan/gobuster` 等工具。\n"
                + "- 若要读取页面/源码用于审计摘要：使用 `curl_request` 并设置 `as_source=true`。\n"
                + "- 若要发送请求/带 Cookie/进行文件上传：使用 `curl_request`；由你决定是否提供 `curl_args` 或使用结构化参数（method/data/json_data/upload_* 等）。\n"
                + "- `curl_request` 返回模式由 `as_source` 控制：as_source=false 时用于正常 HTTP 请求结果摘要。\n"
                + ("\n\n【运行时 Skills（自动注入）】\n" + skills_block if skills_block else "")
            ),
            output_format=(
                "每轮回复必须先承接大Q：收到大Q动作：must_use_tool=...（工具名或空字符串），并说明如何执行；"
                "若反对则必须紧接写【反对理由】与【替代方案】，然后再行动。"
            ),
        )
    )
    worker_messages = [worker_sys] + state["messages"]
    ai_message = _invoke_with_retry(llm_with_tools, worker_messages, "worker_small_d")
    _accumulate_token_usage("worker_small_d", ai_message)
    _print_agent_turn(ai_message, supervisor_content)
    return {"messages": [ai_message], "search_notes": "", "reports_board": board}


def final_llm(state: State, config: RunnableConfig | None = None) -> Any:
    summary_prefix = ""
    if state.get("doc_content"):
        summary_prefix += "【技术文档】\n" + state["doc_content"][:2000] + "\n\n"
    if state.get("code_summary"):
        summary_prefix += "【源码摘要】\n" + state["code_summary"][:1500] + "\n\n"
    sys_msg = SystemMessage(
        content=summary_prefix
        + f"已达最大工具轮数（{MAX_TOOL_ROUNDS}），请直接给最终结论，不要再调用工具。"
    )

    history = list(state["messages"])
    while history and getattr(history[-1], "type", "") == "ai" and getattr(history[-1], "tool_calls", None):
        history.pop()

    model = llm_supervisor if USE_SUPERVISOR else llm_worker
    final_messages = [sys_msg] + history
    if USE_SUPERVISOR:
        final_messages = _to_supervisor_safe_messages(final_messages)
    ai_message = _invoke_with_retry(model, final_messages, "final_llm")
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
        tool = tools_by_name[tool_call["name"]]
        raw_content = tool.invoke(tool_call["args"])
        display_content = raw_content
        if isinstance(raw_content, str) and len(raw_content) > MAX_TOOL_RESULT_CHARS:
            if tool_call["name"] in ("sqlmap_scan", "fenjing_ssti"):
                display_content = _summarize_tool_result(tool_call["name"], raw_content)
                if len(display_content) > MAX_TOOL_RESULT_CHARS:
                    display_content = display_content[:MAX_TOOL_RESULT_CHARS] + "\n\n...(结果已截断)"
            else:
                display_content = raw_content[:MAX_TOOL_RESULT_CHARS] + "\n\n...(结果已截断)"

        if tool_call["name"] == "fetch_ctf_excerpt" and isinstance(display_content, str):
            try:
                payload = display_content
                if payload.startswith("[fetch_ctf_excerpt]"):
                    payload = payload.split("\n", 1)[1] if "\n" in payload else ""
                data = json.loads(payload) if payload.strip().startswith("{") else {}
                chosen_url_val = (data.get("chosen_url") or "").strip()
                excerpt = (data.get("excerpt") or "").strip()
                if excerpt:
                    block = f"来源: {chosen_url_val}\n{excerpt[:5000]}".strip()
                    next_search_notes = (
                        (next_search_notes + "\n\n---\n\n" + block).strip()
                        if next_search_notes
                        else block
                    )
                    next_search_notes = next_search_notes[:10000]
            except Exception:
                pass

        if (
            tool_call["name"] == "curl_request"
            and tool_call.get("args", {}).get("as_source")
            and isinstance(raw_content, str)
            and raw_content.strip()
        ):
            summary = _summarize_source(raw_content)
            accumulated_summary = (
                accumulated_summary + "\n\n---\n\n" + summary if accumulated_summary else summary
            )
        if tool_call["name"] == "read_doc" and isinstance(raw_content, str) and raw_content.strip():
            accumulated_doc = accumulated_doc + "\n\n---\n\n" + raw_content if accumulated_doc else raw_content

        messages.append(
            ToolMessage(
                tool_call_id=tool_call["id"],
                content=json.dumps(display_content, ensure_ascii=False) if not isinstance(display_content, str) else display_content,
                name=tool_call["name"],
            )
        )

    result: State = {"messages": messages, "tool_rounds": state.get("tool_rounds", 0) + 1}
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
        "=========================Author:CUIT-矛盾实验室==========================",
    ]
    deep_green = "\033[38;5;22m"
    reset = "\033[0m"
    banner = "\n".join(art)
    if sys.stdout.isatty():
        print(deep_green + banner + reset, end="")
    else:
        print(banner, end="")


def main() -> None:
    global USE_SUPERVISOR
    global RUN_TOKEN_USAGE
    RUN_TOKEN_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "by_stage": {}}

    parser = argparse.ArgumentParser(description="通用 CTF 渗透测试 Agent（可选大Q监督）")
    parser.add_argument("-t", "--target", required=True, help="目标地址")
    parser.add_argument(
        "--use-supervisor",
        choices=["on", "off"],
        default=os.getenv("USE_SUPERVISOR", "on").lower(),
        help="是否启用本地大Q监督：on/off",
    )
    args = parser.parse_args()
    USE_SUPERVISOR = args.use_supervisor == "on"

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
    print(f"\n[运行模式] 大Q监督: {'开启' if USE_SUPERVISOR else '关闭'}")
    graph.invoke(
        {
            "messages": [
                (
                    "human",
                    f"目标为 {args.target}。尽量拿到 flag 或给出是否存在关键漏洞的最终结论。",
                )
            ]
        },
        config={"recursion_limit": max(200, MAX_TOOL_ROUNDS * 2 + 20)},
    )
    _print_token_usage_summary()


if __name__ == "__main__":
    main()
