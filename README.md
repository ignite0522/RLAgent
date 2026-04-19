# RL Agent 中文

## 依赖安装
`pip install -r requirements.txt`

这是一个基于 LangGraph/ LangChain 的通用 CTF 渗透测试 Agent，支持工具调用闭环；也提供 LoRA（PEFT）+ REINFORCE 的远程训练流程。

## 训练（LoRA + REINFORCE）
1. 启动环境 RPC 服务（默认监听 `0.0.0.0:8010`）：
   运行：`python env_rpc_server.py`
2. 启动本地 OpenAI 兼容推理服务（用于“大Q”/监督，默认 `0.0.0.0:8001`）：
   运行：`python lora/http_server.py`
   如需指定 LoRA 适配器：`PEFT_ADAPTER_PATH=/path/to/adapter python lora/http_server.py`
3. 运行训练：
   `python train_rl_agent_remote.py --env-url http://127.0.0.1:8010 --target <CTF目标URL> --episodes 24`

训练产物：`runs/peft_rl_remote*/ep_*/`（LoRA 权重+分词器），并写入 `metrics.jsonl`。

## 使用（推理/作战）
命令一：`python Agent.py -t <CTF目标URL> --use-supervisor on`（默认开启本地“大Q”监督）
命令二：`python Agent.py -t <CTF目标URL> --use-supervisor off`

也可直接运行 Web/工具图版本：
`python AgentwithWeb.py -t <CTF目标URL>`

> 运行时请通过 `.env` 或环境变量配置远端模型与本地监督服务地址（例如 `DEEPSEEK_API_KEY`、`REMOTE_WORKER_API_BASE`、`LOCAL_LLM_API_BASE` 等）。



# RL Agent English

## Dependency Installation
`pip install -r requirements.txt`

This is a general-purpose CTF penetration testing agent built on LangGraph/LangChain, with closed-loop tool calling. It also provides a remote training pipeline based on LoRA (PEFT) + REINFORCE.

## Training (LoRA + REINFORCE)

1. Start the environment RPC service (listening on `0.0.0.0:8010` by default):
   ```bash
   python env_rpc_server.py
   ```

2. Start the local OpenAI-compatible inference service (used as the "supervisor" / guidance model, listening on `0.0.0.0:8001` by default):
   ```bash
   python lora/http_server.py
   ```
   If you need to specify a LoRA adapter:
   ```bash
   PEFT_ADAPTER_PATH=/path/to/adapter python lora/http_server.py
   ```

3. Run training:
   ```bash
   python train_rl_agent_remote.py --env-url http://127.0.0.1:8010 --target <CTF target URL> --episodes 24
   ```

Training artifacts are saved under `runs/peft_rl_remote*/ep_*/` (LoRA weights + tokenizer), and metrics are written to `metrics.jsonl`.

## Usage (Inference / Operations)

Command 1:
```bash
python Agent.py -t <CTF target URL> --use-supervisor on
```
(Local supervisor enabled by default)

Command 2:
```bash
python Agent.py -t <CTF target URL> --use-supervisor off
```

You can also run the Web / tool-graph version directly:
```bash
python AgentwithWeb.py -t <CTF target URL>
```

> At runtime, configure the remote model and local supervisor service addresses via `.env` or environment variables, such as `DEEPSEEK_API_KEY`, `REMOTE_WORKER_API_BASE`, and `LOCAL_LLM_API_BASE`.
