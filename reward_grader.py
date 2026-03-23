import json
import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

import AgentwithWeb

def _wrap_poml(
    *,
    role: str,
    task: str,
    output_format: str,
) -> str:
    """
    复用 env_rpc_server.py 的 poml 封装协议。

    这里本地实现，避免 reward_grader.py 反向 import env_rpc_server.py
    导致循环依赖。
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


def grade_step_reward(
    *,
    target: str,
    tools_catalog: str,
    report_before: str,
    report_after: str,
    action_tool: str,
    actual_tool: str,
    action_reason: str,
    flag_found: bool,
) -> float:
    """
    使用一个独立的打分 Agent，对“大Q 本轮给的小D工具指导”进行主观评分：
    - 关注点1：对下一轮小D是否产生了正向作用（信息增益、排除假设、推进利用链等）
    - 关注点2：从全局来看，是否有助于更快拿到 flag / 得到关键结论

    评分区间限定在 [-1.0, 1.0]，返回 float；解析失败时回退为 0.0。
    """
    # 这里用确定性规则先兜底，让训练信号更符合系统提示。
    action_tool = (action_tool or "").strip()
    actual_tool = (actual_tool or "").strip()
    if not action_tool:
        # 大Q 没有给出工具指导，本步通常应视为低价值甚至浪费
        return -0.3
    if action_tool and actual_tool and actual_tool != action_tool:
        # 大Q 指定了 A，但小D 实际执行了 B
        return -0.4
    try:
        sys = SystemMessage(
            content=_wrap_poml(
                role="你是一个严谨的评分 Agent。",
                task=(
                    "现在你要评价本轮“大Q给的小D工具指导”是否对渗透测试有正向价值。\n"
                    "你需要综合判断：\n"
                    "1）这次指导选择的工具是否合理、是否被小D采纳执行；大Q给出的原因是否准确有用\n"
                    "2）执行后小D新的汇报（report_after）相比执行前（report_before），"
                    "在发现新证据、验证/排除关键假设、推进拿到 flag 方面，是否有实质进展；\n"
                    "3）从全局完成任务（拿到 flag 或给出可靠结论）的角度看，这一步是正向推进、无明显效果，还是在浪费时间或跑偏。\n"
                    "4)注意：如果大Q作出的动作和原因和小D的不同且远远偏离去拿flag，打出-0.5~-0.4之间的分数"
                    "5)注意：这个细则最重要，如果大Q没有做出动作（must_use_tool为空），打出-0.4～-0.2之间的分数"
                    "6)注意：如果大Q作出的动作和原因和小D的不同但不偏离去拿flag，打出-0.2~-0.1之间的分数，具体分数根据大Q给出的原因是否有用来定。\n"
                    "7)注意：如果大Q作出的动作和原因和小D的相同，且都非常合理且有助于拿flag，打出0.5~1.0之间的分数，具体分数根据大Q给出的原因是否准确有用来定。\n"
                    "你不需要做渗透本身，只对“这一轮动作的价值”进行打分。"
                    "注意：再强调一遍！！这个细则最重要，如果大Q没有做出动作（must_use_tool为空），打出-0.4～-0.2之间的分数"
                ),
                output_format=(
                    "你必须只输出一个 JSON 对象，不能输出多余文本，格式如下：\n"
                    "{\"score\": <介于 -1.0 和 1.0 之间的浮点数>, \"reason\": \"一句话解释原因\"}\n"
                    "score>0 表示明显正向；score≈0 表示几乎没有帮助；score<0 表示有害或明显浪费时间。注意只对大Q给出的的工具和原因进行评分,"
                ),
            )
        )
        human_task = (
            f"【CTF 目标】\n{target}\n\n"
            f"【可用工具列表（节选）】\n{tools_catalog}\n\n"
            f"【本轮大Q动作工具】\n"
            f"must_use_tool = {action_tool!r}\n\n"
            f"【本轮大Q使用原因】\n"
            f"action_reason = {(action_reason or '').strip()!r}\n\n"
            f"【小D 实际执行的动作工具】\n"
            f"actual_tool = {actual_tool!r}\n\n"
            f"【是否已在本轮或之前发现 flag】\n"
            f"flag_found = {flag_found}\n\n"
            f"【本轮执行前，小D 给大Q的历史汇报（report_before）】\n"
            f"{(report_before or '').strip()}\n\n"
            f"【本轮工具执行后，小D 给出的最新汇报（report_after）】\n"
            f"{(report_after or '').strip()}\n\n"
            "请只基于上述信息判断“这一步大Q动作对后续拿到 flag 的价值”，并严格按要求输出 JSON。"
        )
        user = HumanMessage(
            content=_wrap_poml(
                role="评分输入",
                task=human_task,
                output_format="",
            )
        )
        resp = AgentwithWeb.llm_worker.invoke([sys, user])
        text: Any = getattr(resp, "content", "") or ""
        try:
            obj = json.loads(text)
        except Exception:
            # 支持模型把 JSON 包在其它文本里的情况
            m = re.search(r"\{[\s\S]*\}", str(text))
            if not m:
                return 0.0
            obj = json.loads(m.group(0))
        score = float(obj.get("score", 0.0))
        # 显式裁剪到 [-1, 1]
        if score > 1.0:
            score = 1.0
        if score < -1.0:
            score = -1.0
        return score
    except Exception:
        return 0.0


def grade_step_reward_with_raw(
    *,
    target: str,
    tools_catalog: str,
    report_before: str,
    report_after: str,
    action_tool: str,
    actual_tool: str,
    action_reason: str,
    flag_found: bool,
) -> tuple[float, str]:
    """
    和 `grade_step_reward` 相同，但额外返回评分 Agent 的原始输出文本。

    用于在终端逐 step 打印，便于你检查评分是否符合预期。
    """
    action_tool = (action_tool or "").strip()
    actual_tool = (actual_tool or "").strip()
    if not action_tool:
        return -0.3, '{"score": -0.3, "reason": "must_use_tool为空：大Q没有给出工具指导，判为低价值"}'
    if action_tool and actual_tool and actual_tool != action_tool:
        return -0.4, '{"score": -0.4, "reason": "大Q指定工具与小D实际工具不一致，判为低价值"}'
    try:
        sys = SystemMessage(
            content=_wrap_poml(
                role="你是一个严谨的评分 Agent。",
                task=(
                    "现在你要评价本轮“大Q给的小D工具指导”是否对渗透测试有正向价值。\n"
                    "你需要综合判断：\n"
                    "1）这次指导选择的工具是否合理、是否被小D采纳执行；\n"
                    "2）执行后小D新的汇报（report_after）相比执行前（report_before），"
                    "在发现新证据、验证/排除关键假设、推进拿到 flag 方面，是否有实质进展；\n"
                    "3）从全局完成任务（拿到 flag 或给出可靠结论）的角度看，这一步是正向推进、无明显效果，还是在浪费时间或跑偏。\n"
                    "你不需要做渗透本身，只对“这一轮动作的价值”进行打分。"
                ),
                output_format=(
                    "你必须只输出一个 JSON 对象，不能输出多余文本，格式如下：\n"
                    "{\"score\": <介于 -1.0 和 1.0 之间的浮点数>, \"reason\": \"一句话解释原因\"}\n"
                    "score>0 表示明显正向；score≈0 表示几乎没有帮助；score<0 表示有害或明显浪费时间。"
                ),
            )
        )
        human_task = (
            f"【CTF 目标】\n{target}\n\n"
            f"【可用工具列表（节选）】\n{tools_catalog}\n\n"
            f"【本轮大Q动作工具】\n"
            f"must_use_tool = {action_tool!r}\n\n"
            f"【本轮大Q使用原因】\n"
            f"action_reason = {(action_reason or '').strip()!r}\n\n"
            f"【小D 实际执行的动作工具】\n"
            f"actual_tool = {actual_tool!r}\n\n"
            f"【是否已在本轮或之前发现 flag】\n"
            f"flag_found = {flag_found}\n\n"
            f"【本轮执行前，小D 给大Q的历史汇报（report_before）】\n"
            f"{(report_before or '').strip()}\n\n"
            f"【本轮工具执行后，小D 给出的最新汇报（report_after）】\n"
            f"{(report_after or '').strip()}\n\n"
            "请只基于上述信息判断“这一步大Q动作对后续拿到 flag 的价值”，并严格按要求输出 JSON。"
        )
        user = HumanMessage(
            content=_wrap_poml(
                role="评分输入",
                task=human_task,
                output_format="",
            )
        )
        resp = AgentwithWeb.llm_worker.invoke([sys, user])
        raw_text: str = getattr(resp, "content", "") or ""

        try:
            obj = json.loads(raw_text)
        except Exception:
            # 支持模型把 JSON 包在其它文本里的情况
            m = re.search(r"\{[\s\S]*\}", str(raw_text))
            if not m:
                return 0.0, raw_text
            obj = json.loads(m.group(0))

        score = float(obj.get("score", 0.0))
        if score > 1.0:
            score = 1.0
        if score < -1.0:
            score = -1.0
        return score, raw_text
    except Exception:
        return 0.0, ""

