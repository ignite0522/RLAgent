from __future__ import annotations
from pathlib import Path

_SKILL_CACHE: dict[str, str] = {}


def _runtime_skills_dir(base_dir: Path) -> Path:
    return base_dir / "runtime_skills"


def read_runtime_skill(*, base_dir: Path, name: str) -> str:
    """Read one runtime skill markdown by name (cached)."""
    key = (name or "").strip()
    if not key:
        return ""
    if key in _SKILL_CACHE:
        return _SKILL_CACHE[key]

    path = _runtime_skills_dir(base_dir) / f"{key}.md"
    if not path.is_file():
        _SKILL_CACHE[key] = ""
        return ""
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:
        txt = ""
    _SKILL_CACHE[key] = txt
    return txt

def select_runtime_skills(context_text: str) -> list[str]:
    """Pick skill names based on simple keyword heuristics."""
    t = (context_text or "").lower()
    picked: list[str] = []

    if any(
        k in t
        for k in (
            "unserialize",
            "serialize",
            "反序列化",
            "对象注入",
            "pop链",
            "pop 链",
            "__wakeup",
            "__destruct",
        )
    ):
        picked.append("php-unserialize-ctf")

    # 2) 命令执行（RCE / Command Execution）
    # 触发条件：上下文中出现常见“执行命令/远程代码执行”的描述关键词，或危险函数名。
    if any(
        k in t
        for k in (
            "命令执行",
            "远程代码执行",
            "rce",
            "remote code execution",
            "command injection",
            "执行命令",
            "system(",
            "exec(",
            "eval(",
            "popen",
            "passthru",
            "shell_exec",
            "proc_open",
            "subprocess",
            "os.system",
            "child_process",
            "spawn",
            "execfile",
            "runtime.getruntime",
            "processbuilder",
        )
    ):
        picked.append("command-exec-ctf")


    return picked


def build_skills_block(*, base_dir: Path, context_text: str) -> str:
    """Return concatenated content for selected skills."""
    names = select_runtime_skills(context_text)
    blocks: list[str] = []
    for n in names:
        body = read_runtime_skill(base_dir=base_dir, name=n).strip()
        if body:
            # 使用 poml 对 skill 内容做结构化封装，便于提示词解析/约束
            blocks.append(
                "\n".join(
                    [
                        "<skill_poml>",
                        f"<skill_name>{n}</skill_name>",
                        body,
                        "</skill_poml>",
                    ]
                ).strip()
            )
    return "\n\n".join(blocks).strip()

