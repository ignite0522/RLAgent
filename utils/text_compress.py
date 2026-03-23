from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage


StripFn = Callable[[str], str]
InvokeFn = Callable[[list], object]


@dataclass
class RollingTextState:
    summary: str
    items: list[str]


def _safe_strip(strip_fn: StripFn | None, text: str) -> str:
    if not strip_fn:
        return (text or "").strip()
    try:
        return strip_fn(text or "").strip()
    except Exception:
        return (text or "").strip()


def summarize_text(
    *,
    invoke_fn: InvokeFn,
    strip_fn: StripFn | None,
    text: str,
    sys_prompt: str,
    max_input_chars: int = 6000,
) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if len(t) > max_input_chars:
        t = t[-max_input_chars:]
    resp = invoke_fn([SystemMessage(content=sys_prompt), HumanMessage(content=t)])
    content = getattr(resp, "content", "") if resp is not None else ""
    return _safe_strip(strip_fn, str(content))


def compose_rolling_text(*, state: RollingTextState, keep_recent: int = 0) -> str:
    parts: list[str] = []
    if state.summary.strip():
        parts.append("【历史汇报摘要】\n" + state.summary.strip())
    # 当摘要为空时，如果不输出 items，state_text 会直接变成空串，导致大Q/训练端看不到任何状态。
    # keep_recent>0 时输出最近若干条“历轮汇报原文”；若摘要为空且 keep_recent<=0，则至少输出最近 1 条以保证可观测性。
    kept_items: list[str] = []
    if keep_recent and keep_recent > 0:
        kept_items = state.items[-keep_recent:]
    elif not parts and state.items:
        kept_items = state.items[-1:]

    if kept_items:
        parts.append("\n\n====\n\n".join(p for p in kept_items if p).strip())

    return "\n\n====\n\n".join(p for p in parts if p).strip()


def rollup_reports_for_bigq(
    *,
    invoke_fn: InvokeFn,
    strip_fn: StripFn | None,
    summary: str,
    reports: list[str],
    max_chars: int,
    keep_recent: int,
) -> tuple[str, list[str], str]:
    """
    给大Q的 state 做滚动压缩：
    - 目标：返回的 state_text 长度 <= max_chars（尽量）
    - 方法：将所有原文滚动压缩进 summary（不保留原文列表）
    """
    st = RollingTextState(summary=summary or "", items=list(reports or []))
    state_text = compose_rolling_text(state=st, keep_recent=keep_recent)

    sys_prompt = (
        "你是渗透测试记录压缩器。请把下面这些“历轮汇报原文”压缩成不超过10行中文摘要，"
        "只保留关键事实与结论，不要编造，不要提出下一步。"
    )

    # 只要超过阈值，就把当前所有 items 压进 summary，并清空 items（不保留“最近N条原文”）
    while len(state_text) > max_chars and st.items:
        to_summarize = st.items
        st.items = []
        old_text = "\n\n---\n\n".join(to_summarize).strip()
        new_sum = ""
        try:
            new_sum = summarize_text(
                invoke_fn=invoke_fn,
                strip_fn=strip_fn,
                text=old_text,
                sys_prompt=sys_prompt,
                max_input_chars=6000,
            )
        except Exception:
            new_sum = ""

        if new_sum:
            st.summary = (st.summary.strip() + "\n" + new_sum).strip() if st.summary.strip() else new_sum
        else:
            st.summary = (st.summary.strip() + "\n（更早汇报已压缩）").strip()

        state_text = compose_rolling_text(state=st, keep_recent=keep_recent)

    if len(state_text) > max_chars:
        state_text = state_text[-max_chars:]

    return st.summary, st.items, state_text

