import argparse
import json
import os
import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import requests
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:  # 未安装 tqdm 时退化为可迭代包装（避免调用 set_postfix 报错）
    class _TqdmFallback:
        def __init__(self, iterable, **_kwargs):
            self._it = iterable

        def __iter__(self):
            return iter(self._it)

        def set_postfix(self, **_kwargs) -> None:
            pass

        def set_postfix_str(self, _s: str) -> None:
            pass

    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return _TqdmFallback(iterable, **kwargs)


# 用于从模型输出文本中提取 JSON 对象的正则表达式
_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """追加一行 JSON（JSON Lines），供训练过程与 tqdm 同级指标落盘。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _build_chat_inputs(tokenizer: AutoTokenizer, *, system: str, user: str, device: torch.device):
    """
    用模型自带 chat_template 构造对话输入，避免纯文本喂法导致输出发散/乱码。
    将 system + user 拼装成标准 chat 格式，再 tokenize 并移至目标设备。
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # apply_chat_template 会自动加上模型对话格式（如 <|im_start|> 等特殊 token）
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, return_tensors="pt").to(device)


def _extract_tool_from_json(text: str) -> str:
    """
    从模型生成的文本中提取 JSON，读取 must_use_tool 字段。
    若解析失败或字段不存在，返回空字符串（表示"不强制调用任何工具"）。
    """
    if not text:
        return ""
    m = _JSON_RE.search(text)
    if not m:
        return ""
    try:
        obj = json.loads(m.group(0))
        return (obj.get("must_use_tool") or "").strip()
    except Exception:
        return ""


def _extract_tool_and_reason_from_json(text: str) -> tuple[str, str]:
    """
    从模型生成的文本中提取 JSON，读取 must_use_tool 与 reason 字段。
    若解析失败或字段不存在：must_use_tool 返回空字符串，reason 返回空字符串。
    """
    if not text:
        return "", ""
    m = _JSON_RE.search(text)
    if not m:
        return "", ""
    try:
        obj = json.loads(m.group(0))
        tool = (obj.get("must_use_tool") or "").strip()
        reason = (obj.get("reason") or "").strip()
        return tool, reason
    except Exception:
        return "", ""


def _sequence_logprob(model, input_ids: torch.Tensor, gen_ids: torch.Tensor) -> torch.Tensor:
    """
    计算模型在给定 prompt（input_ids）条件下，生成 gen_ids 的对数概率之和
    这是 REINFORCE 算法中计算策略梯度所需的 log π(a|s)
    """
    full = torch.cat([input_ids, gen_ids], dim=-1)
    out = model(full)
    logits = out.logits[:, :-1, :]                         # 去掉最后一个位置（无对应 label）
    labels = full[:, 1:]                                   # 标签为整个序列右移一位
    gen_len = gen_ids.shape[1]
    gen_logits = logits[:, -gen_len:, :]                   # 只取生成部分的 logits
    gen_labels = labels[:, -gen_len:]                      # 只取生成部分的 labels
    logp = torch.log_softmax(gen_logits, dim=-1).gather(-1, gen_labels.unsqueeze(-1)).squeeze(-1)
    return logp.sum(dim=-1)                                # 对 token 维度求和 → 序列 log prob


@dataclass
class Transition:
    input_ids: torch.Tensor  # prompt 的 token IDs
    gen_ids: torch.Tensor    # 模型生成内容的 token IDs
    reward: float            # 环境返回的即时奖励


def main():
    # ── 命令行参数 ────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="服务器端：LoRA + REINFORCE，通过 RPC 环境训练")
    parser.add_argument("--env-url", required=True, help="Mac 环境服务地址，例如 http://mac-ip:8010")
    parser.add_argument("--target", required=True, help="CTF 目标 URL")
    parser.add_argument(
        "--base-model",
        default=os.getenv("BASE_MODEL_PATH", "").strip()
        or "/home/user/.cache/modelscope/hub/models/qwen/Qwen2___5-3B-Instruct",
        help="基座模型路径",
    )
    parser.add_argument("--episodes", type=int, default=24)         # 训练轮数（episode 数量）
    parser.add_argument("--max-steps", type=int, default=40)       # 每轮最多交互步数
    parser.add_argument("--out-dir", default="runs/peft_rl_remote")
    parser.add_argument(
        "--metrics-jsonl",
        default="",
        help="训练指标 JSONL 路径（默认：<out-dir>/metrics.jsonl）",
    )
    parser.add_argument("--lr", type=float, default=2e-4)          # AdamW 学习率
    parser.add_argument("--gamma", type=float, default=1.0)        # 折扣因子（1.0=不折扣）
    parser.add_argument("--temperature", type=float, default=0.8)  # 采样温度（越高越随机）
    parser.add_argument("--top-p", type=float, default=0.95)       # nucleus sampling 阈值
    parser.add_argument("--top-k", type=int, default=0)            # top-k 采样（0=不限制）
    parser.add_argument("--max-new-tokens", type=int, default=748)  # 每步最多生成的 token 数
    parser.add_argument("--quiet", action="store_true", help="关闭每步日志（默认开启）")
    parser.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条（适合重定向日志/无 TTY）")
    parser.add_argument("--max-obs-chars", type=int, default=4000, help="截断小D汇报长度（越小越省显存）")
    parser.add_argument("--parse-fail-penalty", type=float, default=0.0, help="action 解析失败（空）时额外奖励惩罚（默认0）")
    args = parser.parse_args()

    # 每次运行结果都落到唯一目录，避免覆盖旧实验数据
    default_out_dir = "runs/peft_rl_remote"
    run_rand: int | None = None
    if (args.out_dir or "").strip() == default_out_dir:
        run_rand = secrets.randbelow(10_000_000)  # 6~7 位随机数
        args.out_dir = f"{default_out_dir}+{run_rand}"

        # 让训练采样更可追溯（至少对 torch 层面）
        torch.manual_seed(run_rand)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_rand)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_jsonl = (args.metrics_jsonl or "").strip() or os.path.join(args.out_dir, "metrics.jsonl")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _append_jsonl(
        metrics_jsonl,
        {
            "event": "run_start",
            "run_id": run_id,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "env_url": args.env_url,
            "target": args.target,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "gamma": args.gamma,
            "max_new_tokens": args.max_new_tokens,
            "max_obs_chars": args.max_obs_chars,
            "base_model": args.base_model,
            "out_dir": args.out_dir,
            "run_rand": run_rand,
        },
    )
    print(f"[train] metrics_jsonl={metrics_jsonl}", flush=True)

    # ── 设备 & 模型加载 ───────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16   # GPU 用半精度节省显存
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    print(f"[train] device={device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}", flush=True)
    print(f"[train] base_model_path={args.base_model}", flush=True)

    # 加载分词器，若无 pad_token 则用 eos_token 代替（避免 batch padding 报错）
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 加载基座模型（完整权重，后续套 LoRA）
    base = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, dtype=dtype)
    base.to(device)
    try:
        cfg = base.config
        print(
            f"[train] model_id={getattr(cfg,'_name_or_path','')} layers={getattr(cfg,'num_hidden_layers','?')} "
            f"hidden={getattr(cfg,'hidden_size','?')} heads={getattr(cfg,'num_attention_heads','?')}",
            flush=True,
        )
    except Exception:
        pass

    # ── LoRA 配置 ─────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=int(os.getenv("LORA_R", "8")),                          # 低秩矩阵的秩
        lora_alpha=int(os.getenv("LORA_ALPHA", "16")),            # 缩放系数（通常=2r）
        lora_dropout=float(os.getenv("LORA_DROPOUT", "0.05")),    # dropout 防过拟合
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=os.getenv("LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj").split(","),
    )
    model = get_peft_model(base, lora_cfg)

    # 关闭 dropout（eval 模式），保证采样时策略分布稳定；LoRA 参数仍可反传梯度
    model.eval()

    # 显存优化：关闭 KV cache（训练时无法复用），开启梯度检查点（用算力换显存）
    try:
        model.config.use_cache = False
    except Exception:
        pass
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # ── 优化器 & 基线初始化 ───────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # baseline：REINFORCE 方差减小技巧，用指数移动平均估计当前期望回报
    # 梯度更新公式：∇J ≈ (G_t - baseline) · ∇log π(a|s)
    baseline = 0.0
    baseline_momentum = 0.9  # EMA 动量（越大=基线越"惰性"）

    # ════════════════════════════════════════════════════════════════════════════
    # 主训练循环：每个 episode = 一次完整的 CTF 攻击尝试
    # ════════════════════════════════════════════════════════════════════════════
    # 读取小D的完整 system_prompt 规则，直接拼给大Q（你要求的“全量拼进去”）。
    tech_docs_path = Path(__file__).resolve().parent / "tech_docs" / "system_prompt.txt"
    try:
        tech_docs_text = tech_docs_path.read_text(encoding="utf-8")
    except Exception:
        tech_docs_text = ""

    show_progress = not args.no_progress
    ep_pbar = tqdm(
        range(1, args.episodes + 1),
        desc="Episodes",
        unit="ep",
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for ep in ep_pbar:
        t_ep0 = time.perf_counter()

        # ── Step 0：重置远端环境（Mac 侧的 CTF 沙箱）─────────────────────────
        try:
            r = requests.post(
                f"{args.env_url.rstrip('/')}/reset",
                json={"target": args.target},
                timeout=240,
            )
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            base = args.env_url.rstrip("/")
            raise RuntimeError(
                "无法连接环境服务，请检查以下项：\n"
                f"1) env_rpc_server 是否已启动并监听（当前 env_url={base}）\n"
                "2) 若训练在远端机器运行，env_url 不能用 127.0.0.1（它只指向远端自身），请改成环境服务所在机器的可达 IP\n"
                "3) 端口 8010 是否放通、防火墙/安全组是否允许访问\n"
                f"原始错误：{type(e).__name__}: {e}"
            ) from e
        payload = r.json()
        env_id = payload["env_id"]              # 本轮环境实例 ID
        obs = payload.get("state") or ""        # 初始观测（小D的第一次汇报）
        tools_catalog = payload.get("tools_catalog") or ""  # 可用工具列表

        transitions: list[Transition] = []
        total_reward = 0.0

        _append_jsonl(
            metrics_jsonl,
            {
                "event": "episode_start",
                "run_id": run_id,
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "episode": ep,
                "episodes_total": args.episodes,
                "max_steps": args.max_steps,
                "env_id": env_id,
            },
        )

        # ── 交互循环：每步"观测→生成动作→环境执行→得到奖励"────────────────────
        step_pbar = tqdm(
            range(args.max_steps),
            desc=f"  ep{ep} steps",
            unit="step",
            leave=False,
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for step_idx in step_pbar:
            t_step0 = time.perf_counter()

            # 截断观测，防止 prompt 过长撑爆显存
            obs_short = obs if len(obs) <= args.max_obs_chars else obs[-args.max_obs_chars :]

            # 构造 system prompt：定义大Q（本模型）的角色与输出格式
            system_prompt = (
                f"{tech_docs_text}\n"
                "\n"
                "你是本地大模型大Q，负责参考远端小模型小D的汇报，然后给他下达下一步'动作选择'（只选工具，不写长篇建议）。\n"
                "你是一名资深网络安全与CTF专家，熟悉常见 Web 漏洞（如 SQL 注入、XSS、SSRF、命令执行等）的利用与测试流程。\n"
                "在选择工具时，要综合效率与价值：优先选择能验证关键假设或推进获取 flag 的动作，避免明显重复或低收益的探测。\n"
                "\n"
                "【输出格式｜只能输出 JSON】\n"
                "{\"must_use_tool\": \"<工具名或空字符串>\", \"reason\": \"<必须使用时只给一句原因；must_use_tool为空时可为空或只给一句简短原因>\"}\n"
                "- must_use_tool 为空字符串时，表示本轮不强制调用工具（例如需要小D先做纯分析/总结）。\n"
                "【重要约束】must_use_tool 必须严格从'工具列表'里选择一个名字；如果不确定，就返回空字符串。\n"
                "\n"
                "大Q的输出内容只包含：选哪个工具（must_use_tool）以及为什么选它（reason），以及不输出其它文本；其余分析交给小D。\n"
            )
            # 构造 user prompt：将当前可用工具和小D的情况汇报注入
            user_prompt = (
                f"【工具列表（名称:用途）】\n{tools_catalog}\n\n"
                f"【小D 情况汇报】\n{obs_short}\n"
            )
            inputs = _build_chat_inputs(tok, system=system_prompt, user=user_prompt, device=device)

            # ── 策略采样（on-policy）：用当前 LoRA 模型生成动作 ──────────────
            # torch.no_grad() 避免生成阶段建立计算图（节省显存），梯度在事后重算
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    do_sample=True,                            # 采样而非贪心，保证探索
                    temperature=max(args.temperature, 1e-6),   # 温度控制随机性
                    top_p=args.top_p,                          # nucleus sampling
                    **({"top_k": args.top_k} if args.top_k > 0 else {}),
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            t_after_gen = time.perf_counter()
            # gen 包含 prompt + 生成内容，切片取生成部分
            gen_ids = gen[:, inputs["input_ids"].shape[1] :]
            text = tok.decode(gen_ids[0], skip_special_tokens=True)

            # 从生成文本中解析出工具名（即"动作"）
            action, action_reason = _extract_tool_and_reason_from_json(text)

            # ── 与远端环境交互：执行动作，得到新状态和奖励 ────────────────────
            s = requests.post(
                f"{args.env_url.rstrip('/')}/step",
                json={"env_id": env_id, "action_tool": action, "action_reason": action_reason},
                timeout=480,
            )
            s.raise_for_status()
            t_after_env = time.perf_counter()
            step = s.json()
            reward = float(step["reward"])          # 本步奖励
            done = bool(step.get("done"))           # 是否达到终止条件（找到 flag 或超时）
            obs = step.get("state") or ""           # 下一步观测（小D新的汇报）
            # 对话日志由 Mac 端 env_rpc_server 负责打印，训练端不重复输出

            # 解析失败时施加额外惩罚（默认为0，可通过参数调整）
            if not action:
                reward += float(args.parse_fail_penalty)
            total_reward += reward

            gen_sec = float(t_after_gen - t_step0)
            env_sec = float(t_after_env - t_after_gen)
            step_wall_sec = float(t_after_env - t_step0)

            _append_jsonl(
                metrics_jsonl,
                {
                    "event": "step",
                    "run_id": run_id,
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "episode": ep,
                    "episodes_total": args.episodes,
                    "step_index": step_idx,
                    "max_steps": args.max_steps,
                    "env_id": env_id,
                    "must_use_tool": action,
                    "reason": action_reason,
                    "reward": reward,
                    "total_reward": total_reward,
                    "done": done,
                    "accepted": bool(step.get("accepted")),
                    "actual_tool": step.get("actual_tool", ""),
                    "flag_found": bool(step.get("flag_found")),
                    "flag_snippet": (step.get("flag_snippet") or "")[:200],
                    "gen_sec": round(gen_sec, 4),
                    "env_rpc_sec": round(env_sec, 4),
                    "step_wall_sec": round(step_wall_sec, 4),
                    # 与 tqdm 中 R / last 对应
                    "tqdm_R": round(total_reward, 2),
                    "tqdm_last_reward": round(reward, 2),
                },
            )

            # 只保存 token IDs 到 CPU，不保留计算图，避免显存线性增长
            transitions.append(
                Transition(
                    input_ids=inputs["input_ids"].detach().to("cpu"),
                    gen_ids=gen_ids.detach().to("cpu"),
                    reward=reward,
                )
            )

            if show_progress:
                step_pbar.set_postfix(R=f"{total_reward:.2f}", last=f"{reward:.2f}")
            if done:
                break   # episode 结束，跳出交互循环

        # ════════════════════════════════════════════════════════════════════════
        # REINFORCE 策略梯度更新（episode 结束后统一反传）
        # ════════════════════════════════════════════════════════════════════════

        # ── 计算每步的折扣累积回报 G_t ────────────────────────────────────────
        # 从后往前倒序累加
        returns: list[float] = []
        G = 0.0
        for tr in reversed(transitions):
            G = tr.reward + args.gamma * G
            returns.append(G)
        returns = list(reversed(returns))  # 还原为正序（与 transitions 对齐）

        # 更新baseline：用本轮回报更新 EMA baseline，作为下轮的基线估计
        ep_return = returns[0] if returns else total_reward
        baseline_prev = baseline
        baseline_new = baseline_momentum * baseline_prev + (1 - baseline_momentum) * float(ep_return)

        loss = 0.0
        opt.zero_grad(set_to_none=True)
        if transitions:
            adv_raw = [float(Gt - baseline_prev) for Gt in returns]
            mean_adv = sum(adv_raw) / len(adv_raw)
            var_adv = sum((a - mean_adv) ** 2 for a in adv_raw) / max(len(adv_raw), 1)
            std_adv = var_adv ** 0.5
        else:
            adv_raw = []
            mean_adv = 0.0
            std_adv = 0.0

        for idx, (tr, Gt) in enumerate(zip(transitions, returns)):
            adv = adv_raw[idx]
            if std_adv > 1e-6:
                adv = (adv - mean_adv) / std_adv
            else:
                adv = 0.0
            input_ids = tr.input_ids.to(device)
            gen_ids2 = tr.gen_ids.to(device)
            logp = _sequence_logprob(model, input_ids, gen_ids2)[0]
            loss_step = -logp * adv  # loss = - log π(a|s) * advantage
            loss = loss + float(loss_step.detach().cpu())
            loss_step.backward()
        opt.step()
        baseline = baseline_new                      # 更新 EMA baseline（用于下一 episode）

        ep_wall_sec = float(time.perf_counter() - t_ep0)
        save_path = os.path.join(args.out_dir, f"ep_{ep}")

        if show_progress:
            ep_pbar.set_postfix(R=f"{total_reward:.3f}", n=len(transitions))
        else:
            print(f"[ep {ep}] total_reward={total_reward:.3f} steps={len(transitions)}", flush=True)

        _append_jsonl(
            metrics_jsonl,
            {
                "event": "episode_end",
                "run_id": run_id,
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "episode": ep,
                "episodes_total": args.episodes,
                "total_reward": total_reward,
                "steps": len(transitions),
                "ep_return": float(ep_return),
                "baseline_after": float(baseline),
                "episode_wall_sec": round(ep_wall_sec, 4),
                "save_path": save_path,
                # 与 tqdm Episodes 行 R / n 一致
                "tqdm_R": round(total_reward, 3),
                "tqdm_n": len(transitions),
            },
        )

        # ── 保存本轮 LoRA 权重 ────────────────────────────────────────────────
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)    # 只保存 LoRA delta 权重（体积小）
        tok.save_pretrained(save_path)      # 同时保存分词器，方便后续加载推理

    _append_jsonl(
        metrics_jsonl,
        {
            "event": "run_end",
            "run_id": run_id,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "episodes_done": args.episodes,
            "out_dir": args.out_dir,
        },
    )


if __name__ == "__main__":
    main()