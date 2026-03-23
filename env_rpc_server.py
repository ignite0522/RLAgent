import os
import uuid
import json
import re
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import uvicorn
import AgentwithWeb
from pathlib import Path
from runtime_skills_loader import build_skills_block
from reward_grader import grade_step_reward_with_raw


# 用于从工具输出中检测 flag 的正则表达式，支持 flag{...} 和 ctfshow{...} 两种格式
_FLAG_RE = re.compile(r"(flag\{[^}\n]{0,120}\}|ctfshow\{[^}\n]{0,120}\})", re.IGNORECASE)

# 与 AgentwithWeb.py 对齐：小D历史汇报板做滚动裁剪，防止 state/上下文无限增长
# 大Q state 只保留最近 7 条汇报
_REPORTS_BOARD_KEEP_LAST = 7
_REPORTS_BOARD_MAX_TOTAL_CHARS = 22_000

# 历史小D汇报之间的分隔符（用于避免“多轮汇报看起来直接追加在后面”）
_REPORTS_DIVIDER = "\n\n【===== 小D 汇报分隔 =====】\n\n"


def _trim_reports_board(
    board: Optional[List[str]],
    *,
    keep_last: int = _REPORTS_BOARD_KEEP_LAST,
    max_total_chars: int = _REPORTS_BOARD_MAX_TOTAL_CHARS,
) -> List[str]:
    items = [str(x or "").strip() for x in (board or [])]
    items = [x for x in items if x]
    if keep_last and keep_last > 0:
        items = items[-keep_last:]
    # 先按“条数”截断，再按“总字符数”从最旧开始弹出
    while items and sum(len(x) for x in items) > max_total_chars:
        items.pop(0)
    return items


def _wrap_poml(
    *,
    role: str,
    task: str,
    output_format: str,
) -> str:
    """
    将小D的提示词按统一的 poml 协议封装。

    注意：这里仅做文本封装，不改变上层调用方式。
    """
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


# ── 请求 / 响应数据模型（FastAPI 用 Pydantic 做自动校验）────────────────────────

class ResetReq(BaseModel):
    target: str                 # 要攻击的 CTF 目标 URL


class ResetResp(BaseModel):
    env_id: str                 # 本次 episode 的唯一标识
    state: str                  # 小D的初始汇报（训练端第一步观测）
    tools_catalog: str          # 可用工具列表文本


class StepReq(BaseModel):
    env_id: str
    action_tool: str = ""       # 大Q选择的工具名（可为空字符串）
    action_reason: str = ""    # 大Q给出的使用原因（可为空字符串）


class StepResp(BaseModel):
    env_id: str
    reward: float               # 本步奖励
    accepted: bool              # 小D是否执行了大Q指定的工具
    actual_tool: str            # 小D实际调用的工具名
    flag_found: bool            # 本步是否发现 flag
    flag_snippet: str           # flag 内容片段（发现时才有值）
    state: str                  # 下一步观测（小D新的汇报）
    done: bool                  # episode 是否结束


class _Session:
    """
    维护单次 episode 的完整上下文。
    每次 /reset 创建一个新 Session，/step 在其上持续追加消息。
    """
    def __init__(self, target: str):
        self.target = target
        # 对话历史，初始化时注入攻击任务描述
        self.messages: List[Any] = [
            (
                "human",
                f"目标为 {target}。尽量拿到 flag 或给出是否存在关键漏洞的最终结论。",
            )
        ]
        self.reports_board: List[str] = []   # （预留）历史汇报存档
        # 对过长历史做滚动压缩，summary + 最近若干条原文 = “所有汇报的累加（可压缩）”
        self.reports_summary: str = ""
        # 小D对话历史的滚动摘要（用于控制小D上下文长度）
        self.dialog_summary: str = ""
        self.tool_rounds: int = 0            # 已执行的工具调用轮数（用于判断超时）
        self.done: bool = False              # episode 是否已结束
        # 缓存上一次给训练端的 state，避免重复调用 LLM 产生漂移
        self.last_state: str = ""


# 全局 Session 字典：env_id → _Session
SESSIONS: Dict[str, _Session] = {}


def _tool_catalog() -> str:
    """获取当前可用工具的目录文本（由 AgentwithWeb 模块维护）。"""
    return AgentwithWeb._tool_catalog_text()


def _make_report(messages: List[Any], tools_catalog: str, reports_board: Optional[List[str]] = None) -> str:
    """
    调用大模型做“汇总式报告”，但报告只能基于 ToolMessage 中的真实证据。
    防止无中生有：若证据不足，必须写“未获得证据/未知”，且禁止给出下一步工具选择。
    """
    # 1) 提取目标（来自 reset 阶段的 human 文本）
    target = "（未提供）"
    for m in messages:
        if isinstance(m, tuple) and len(m) >= 2 and m[0] == "human":
            text = str(m[1])
            if "目标为" in text:
                target = text.split("目标为", 1)[1].split("。", 1)[0].strip()
            break

    # 2) 提取 ToolMessage 作为“证据”（改为全量，避免只看最近导致信息丢失）
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    evidence: str
    if not tool_msgs:
        evidence = "【证据】未执行任何工具（ToolMessage 数=0）。"
    else:
        recent = tool_msgs  # 全量工具结果作为证据
        ev_lines: list[str] = []
        for tm in recent:
            name = getattr(tm, "name", "") or "unknown_tool"
            call_id = getattr(tm, "tool_call_id", "") or ""
            content = getattr(tm, "content", "") or ""
            content = " ".join(str(content).splitlines()).strip()
            if call_id:
                ev_lines.append(f"- tool={name} tool_call_id={call_id} content={content}")
            else:
                ev_lines.append(f"- tool={name} content={content}")
        evidence = "【证据(工具全量)】\n" + "\n".join(ev_lines)

    # 3) 把历史汇报板一起注入，避免“上一轮结论丢失”
    board = _trim_reports_board(reports_board)
    if board:
        report_lines: list[str] = []
        for i, rep in enumerate(board):
            rep_str = str(rep or "").strip()
            if not rep_str:
                continue
            report_lines.append(f"- history_report[{i}] {rep_str}")
        if report_lines:
            evidence = evidence + "\n\n【历史小D汇报板(全量)】\n" + "\n".join(report_lines)

    reporter_sys = SystemMessage(
        content=_wrap_poml(
            role="你是大模型小D。",
            task=(
                "负责把“证据”汇总成给大Q的状态汇报。\n"
                "你只能总结你收到的证据中的事实，严禁编造。\n"
                "\n"
                "硬性规则：\n"
                "- 只能基于证据中出现过的内容写“已执行/已发现/已看到”。\n"
                "- 若证据无法支持某结论，必须写“未获得证据/未知”。\n"
                "- 禁止主动给出下一步工具选择。"
            ),
            output_format=(
                "1）当前进展/发现：最少3 条要点；每条末尾必须标注证据来源：`tool=<工具名>` 或 `history_report`。\n"
                "2）当前遇到的困难或疑点：简要说明未知点/需要确认的地方。"
            ),
        )
    )
    reporter_user = HumanMessage(
        content=evidence
        + "\n\n请严格按输出格式汇总。不要加入证据之外的任何尝试或结果。"
    )

    resp = AgentwithWeb.llm_worker.invoke([reporter_sys, reporter_user])
    report = AgentwithWeb._strip_dsml_blocks(getattr(resp, "content", "") or "").strip()
    if report:
        return report

    # 不再提供兜底；让上层直接看到“模型未输出汇报”的真实结果
    return ""


def _build_worker_sys(*, tools_catalog: str, supervisor_content: str, context_text: str) -> SystemMessage:
    """
    构造小D（llm_with_tools）的 system prompt，核心逻辑：
    1. 加载 system_prompt.txt（小D的基础角色设定）
    2. 注入工具列表
    3. 将大Q的动作（must_use_tool）原文透传，要求小D优先执行
    4. 注入运行时 Skills（动态上下文技巧）
    """
    # 读取 system_prompt.txt（与 AgentwithWeb 一致）
    try:
        base_dir = os.path.dirname(AgentwithWeb.__file__)
        sys_prompt_path = os.path.join(base_dir, "tech_docs", "system_prompt.txt")
        with open(sys_prompt_path, "r", encoding="utf-8") as f:
            system_prompt_body = f.read()
    except Exception:
        system_prompt_body = "你是一名 CTF 竞赛与渗透测试专家，负责综合分析并利用题目中的漏洞。"

    base_dir = Path(__file__).resolve().parent
    # 根据当前上下文动态选取相关 skills（如 SQL 注入技巧、文件包含技巧等）
    skills_block = build_skills_block(base_dir=base_dir, context_text=context_text)
    worker_sys_text = (
        system_prompt_body
        + "\n\n【工具列表（重要）】\n"
        + tools_catalog
        + "\n"
        + "【本地大模型大Q最新动作】大Q输出一个 JSON 动作，至少包含：{\"must_use_tool\": \"<工具名或空字符串>\"}（可选包含 reason）。\n"
        + "你的最高优先级是：先解析并承接该 JSON 中的 must_use_tool，然后优先按该工具行动。\n"
        + "若你判断该动作在当前环境下会导致明显无效/浪费/风险（例如工具不可用、参数缺失无法补全、与目标阶段完全不匹配），才允许反对，但必须按下面格式说明理由与替代动作。\n"
        + "大Q 的动作和原因如下（原文照抄，便于你解析）：\n"
        + f"{supervisor_content}\n\n"
        + "你现在是远端小模型小D，职责是：在保证安全与合规的前提下，尽量高效推进渗透测试，并尊重大Q的指导。"
        "如你与大Q观点不同，可以提出，但必须给出清晰理由与替代方案。\n"
        "\n"
        "【协议策略】默认优先尝试 HTTPS；若 HTTPS 因证书/握手/连接失败导致无法获取有效响应，请立即改用 HTTP 继续验证与测试，并在回复中说明原因。\n"
        "\n"
        "【对话格式｜必须先承接大Q】\n"
        "你每一轮回复开头必须先用 1-2 句承接大Q：\n"
        "1）用'收到大Q动作：must_use_tool=... '明确写出你解析到的工具名（或空字符串）；\n"
        "2）说明你将如何执行以及接下来选择什么工具继续利用\n"
        "如果你决定不按大Q建议执行，必须紧接着写【反对理由】与【替代方案】，然后再行动。\n"
        "\n"
        "【硬性约束】\n"
        "- 你只能调用工具列表中的工具；不得编造工具名或自行假设存在 `post_http/whatweb_scan/gobuster` 等工具。\n"
        "- 若要读取页面/源码用于审计摘要：使用 `curl_request` 并设置 `as_source=true`。\n"
        "- 若要发送请求/带 Cookie/进行文件上传：使用 `curl_request`；由你决定是否提供 `curl_args` 或使用结构化参数（method/data/json_data/upload_* 等）。\n"
        "- `curl_request` 返回模式由 `as_source` 控制：as_source=false 时用于正常 HTTP 请求结果摘要。\n"
        + ("\n\n【运行时 Skills（自动注入）】\n" + skills_block + "\n" if skills_block else "")
    )

    return SystemMessage(
        content=_wrap_poml(
            role="你现在是远端小模型小D。",
            task=worker_sys_text,
            output_format=(
                "每轮回复必须先承接大Q：收到大Q动作：must_use_tool=...（工具名或空字符串），并说明如何执行；若反对则必须紧接写【反对理由】与【替代方案】，然后再行动。"
            ),
        )
    )


def _execute_tools(tool_calls: list[dict]) -> tuple[list[ToolMessage], bool, str]:
    """
    执行小D发起的所有工具调用，收集结果并检测 flag。
    返回值：(工具消息列表, 是否发现flag, flag片段)
    """
    tools_by_name = {t.name: t for t in AgentwithWeb.tools}
    tool_messages: list[ToolMessage] = []
    flag_found = False
    flag_snippet = ""

    for tc in tool_calls or []:
        name = tc.get("name", "")
        args = tc.get("args", {}) or {}
        tool = tools_by_name.get(name)
        if tool is None:
            # 工具名不存在时，返回错误信息，不执行
            tool_messages.append(
                ToolMessage(tool_call_id=tc.get("id", ""), name=name, content=f"unknown_tool:{name}")
            )
            continue
        try:
            raw = tool.invoke(args)   # 实际调用工具
        except Exception as e:
            # 工具参数校验失败等情况，不要让整个 /step 崩成 500
            tool_messages.append(
                ToolMessage(
                    tool_call_id=tc.get("id", ""),
                    name=name,
                    content=f"tool_invoke_failed:{name}:{type(e).__name__}:{str(e)}",
                )
            )
            continue

        # 在工具返回结果中扫描 flag（始终在原始文本上做）
        if isinstance(raw, str) and not flag_found:
            m = _FLAG_RE.search(raw)
            if m:
                flag_found = True
                flag_snippet = m.group(1)[:140]

        # 为后续对话准备的内容：
        # 针对 curl_request：按你的要求“不做摘要”，直接把原文塞进 ToolMessage，
        # 由工具本身的 max_chars 参数来限制体积。
        display_content: Any = raw
        if isinstance(raw, str):
            try:
                if name == "curl_request":
                    # 直接返回原文（不走 AgentwithWeb 摘要器）
                    display_content = raw
                elif name in ("sqlmap_scan", "fenjing_ssti"):
                    # SQLmap / SSTI 工具结果摘要
                    display_content = AgentwithWeb._summarize_tool_result(name, raw)
            except Exception:
                # 摘要失败时退回原始内容（后续会再由上层做截断）
                display_content = raw
        tool_messages.append(
            ToolMessage(
                tool_call_id=tc.get("id", ""),
                name=name,
                # 非字符串结果序列化为 JSON，便于后续消息统一处理
                content=json.dumps(display_content, ensure_ascii=False)
                if not isinstance(display_content, str)
                else display_content,
            )
        )
    return tool_messages, flag_found, flag_snippet


# ── FastAPI 应用初始化 ─────────────────────────────────────────────────────────
app = FastAPI(title="CTF Env RPC Server", version="0.1")

@app.post("/reset", response_model=ResetResp)
def reset(req: ResetReq) -> ResetResp:
    """
    /reset：开启新的 episode。
    1. 创建新 Session，生成唯一 env_id
    2. 让小D产出初始汇报作为第一步 state
    3. 返回 env_id + state + tools_catalog 给训练端（大Q）
    """
    env_id = str(uuid.uuid4())
    sess = _Session(req.target.strip())
    SESSIONS[env_id] = sess

    tools_catalog = _tool_catalog()
    report = _make_report(sess.messages, tools_catalog, reports_board=sess.reports_board)
    # 初始化历史汇报板：第一条汇报
    sess.reports_board = _trim_reports_board([report] if report else [])
    sess.reports_summary = ""
    sess.dialog_summary = ""
    # 对大Q而言，state 始终是“截至当前的所有汇报累加”文本
    cumulative_state = _REPORTS_DIVIDER.join(sess.reports_board).strip()
    sess.last_state = cumulative_state
    return ResetResp(env_id=env_id, state=cumulative_state, tools_catalog=tools_catalog)


@app.post("/step", response_model=StepResp)
def step(req: StepReq) -> StepResp:
    """
    /step：执行一步交互，对应训练端的"动作→环境→奖励"循环。

    完整流程：
    1. 大Q传入 action_tool（工具名），构造小D的 system prompt
    2. 调用 llm_with_tools（小D）生成回复，可能含工具调用
    3. 执行小D发起的工具调用，检测 flag
    4. 计算本步奖励（接受大Q建议+0.2，拒绝-0.1，发现flag+2.0）
    5. 让小D重新汇报，作为下一步 state 返回给训练端
    """
    sess = SESSIONS.get(req.env_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="env_id_not_found")
    # 若 episode 已结束，直接返回 done，不再执行任何操作
    if sess.done:
        return StepResp(
            env_id=req.env_id,
            reward=0.0,
            accepted=False,
            actual_tool="",
            flag_found=False,
            flag_snippet="",
            state="done",
            done=True,
        )

    tools_catalog = _tool_catalog()
    action_tool = (req.action_tool or "").strip()
    action_reason = (req.action_reason or "").strip()
    # 将大Q的动作序列化为 JSON 字符串，传给小D
    supervisor_obj: dict[str, str] = {"must_use_tool": action_tool}
    if action_reason:
        supervisor_obj["reason"] = action_reason
    supervisor_content = json.dumps(supervisor_obj, ensure_ascii=False)

    # 用上一步缓存的汇报作为大Q的 state，避免重复调用 LLM 导致漂移
    report_for_q = sess.last_state or ""


    # 构造小D本轮的 system prompt（注入大Q动作 + 工具列表 + skills）
    worker_sys = _build_worker_sys(
        tools_catalog=tools_catalog,
        supervisor_content=supervisor_content,
        context_text=(report_for_q + "\n\n" + supervisor_content).strip(),
    )
    # 小D根据完整对话历史 + system prompt 生成本轮回复
    ai_msg = AgentwithWeb.llm_with_tools.invoke([worker_sys] + sess.messages)
    sess.messages.append(ai_msg)   # 将小D回复追加到对话历史

    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    actual_tool = tool_calls[0].get("name", "") if tool_calls else ""
    accepted = bool(action_tool and actual_tool == action_tool)

    # 执行工具调用，收集结果，检测 flag
    tool_msgs, flag_found, flag_snippet = _execute_tools(tool_calls)
    sess.messages.extend(tool_msgs)
    sess.tool_rounds += 1

    # ── 步数惩罚（让“越拖越久”变差）─────────────────────────────────────────
    # 随着 step 越接近 max_rounds，惩罚线性增大。
    # 通过环境变量调强度，避免手工改代码。
    max_rounds = int(os.getenv("ENV_MAX_ROUNDS", "40"))
    if max_rounds > 0:
        step_norm = min(1.0, float(sess.tool_rounds) / float(max_rounds))
    else:
        step_norm = 0.0
    step_count_penalty_max = float(os.getenv("ENV_STEP_COUNT_PENALTY_MAX", "0.3"))
    step_count_penalty = -step_count_penalty_max * step_norm

    # ── 奖励设计（基于评分 Agent）───────────────────────────────────────────────
    reward = 0.0
    reward += step_count_penalty
    if flag_found:
        reward += 7.0        # 发现 flag 给予大额奖励（主目标）
        sess.done = True

    # 复用同一次生成的 next_report：
    next_report = _make_report(sess.messages, tools_catalog, reports_board=sess.reports_board)

    if not sess.done:
        step_score = 0.0
        grader_raw = ""
        step_score, grader_raw = grade_step_reward_with_raw(
            target=sess.target,
            tools_catalog=tools_catalog,
            report_before=report_for_q,
            report_after=next_report or "",
            action_tool=action_tool,
            actual_tool=actual_tool,
            action_reason=action_reason,
            flag_found=flag_found,
        )
        reward += float(step_score)

    # 让小D重新汇报，生成下一步汇报；并与历史汇报累加，供大Q作为下一次观测
    if next_report:
        sess.reports_board.append(next_report)
    sess.reports_board = _trim_reports_board(sess.reports_board)

    # 大Q看到的小D汇报：直接累加原始汇报（禁用滚动压缩，避免丢信息）
    cumulative_state = _REPORTS_DIVIDER.join(sess.reports_board).strip()
    sess.reports_summary = ""
    sess.last_state = cumulative_state

    # 超出最大轮数则强制结束 episode（防止无限循环）
    if sess.tool_rounds >= max_rounds:
        sess.done = True

    resp = StepResp(
        env_id=req.env_id,
        reward=reward,
        accepted=accepted,
        actual_tool=actual_tool,
        flag_found=flag_found,
        flag_snippet=flag_snippet,
        state=cumulative_state,
        done=sess.done,
    )

    # ── 彩色终端日志────────────────
    try:
        BLUE = "\033[34m"
        PURPLE = "\033[35m"
        GREEN = "\033[32m"
        RED = "\033[31m"
        RESET = "\033[0m"
        print(f"\n{GREEN}[小D汇报]:{RESET}")
        report_txt = str(report_for_q or "").strip()
        if report_txt:
            # 按分隔符输出每一轮汇报，保持换行可读性
            blocks = [b.strip() for b in report_txt.split(_REPORTS_DIVIDER) if b.strip()]
            for i, b in enumerate(blocks, start=1):
                print(f"\n【汇报 {i}/{len(blocks)}】\n{b}")
        else:
            print("（空）")
        print(f"\n{BLUE}[大Q]:{RESET}")
        print(supervisor_content)
        print(f"{BLUE}[小D]:{RESET}")
        content_str = getattr(ai_msg, "content", "") or ""
        compact = " ".join(line.strip() for line in str(content_str).splitlines() if line.strip())
        if not compact and tool_calls:
            compact = "（本轮小D未输出文本，仅发起了工具调用）"
        print(compact)
        if tool_calls:
            print(f"{PURPLE}[tool_calls]:{RESET}")
            for tc in tool_calls:
                print(f"- {tc.get('name')} args={tc.get('args')}\n")
        if not sess.done:
            # 打印评分 Agent 的原始 JSON 输出，便于检查奖励是否“符合预期”
            try:
                if grader_raw:
                    print(f"{PURPLE}[reward_grader_raw]:{RESET} {grader_raw.strip()}")
                else:
                    print(f"{PURPLE}[reward_grader_raw]:{RESET} （空输出）")
            except Exception:
                pass
        if flag_found:
            print(f"{RED}[flag_found]{RESET} {flag_snippet}", flush=True)
    except Exception:
        pass

    return resp


if __name__ == "__main__":
    # 从环境变量读取监听地址和端口，默认 0.0.0.0:8010
    host = os.getenv("ENV_HOST", "0.0.0.0")
    port = int(os.getenv("ENV_PORT", "8010"))
    uvicorn.run(app, host=host, port=port)