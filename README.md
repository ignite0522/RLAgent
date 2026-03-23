# LangGraph CTF Agent

## 依赖安装
`pip install -r requirements.txt`

这是一个基于 LangGraph/ LangChain 的通用 CTF 渗透测试 Agent，支持工具调用闭环；也提供 LoRA（PEFT）+ REINFORCE 的远程训练流程。

## 训练（LoRA + REINFORCE）
1. 启动环境 RPC 服务（默认监听 `0.0.0.0:8010`）：
   - `python env_rpc_server.py`
2. 启动本地 OpenAI 兼容推理服务（用于“大Q”/监督，默认 `0.0.0.0:8001`）：
   - `python lora/http_server.py`
   - 如需指定 LoRA 适配器：`PEFT_ADAPTER_PATH=/path/to/adapter python lora/http_server.py`
3. 运行训练：
   - `python train_rl_agent_remote.py --env-url http://127.0.0.1:8010 --target <CTF目标URL> --episodes 24`

训练产物：`runs/peft_rl_remote*/ep_*/`（LoRA 权重+分词器），并写入 `metrics.jsonl`。

## 使用（推理/作战）
- `python Agent.py -t <CTF目标URL> --use-supervisor on`（默认开启本地“大Q”监督）
- `python Agent.py -t <CTF目标URL> --use-supervisor off`

也可直接运行 Web/工具图版本：
- `python AgentwithWeb.py -t <CTF目标URL>`

> 运行时请通过 `.env` 或环境变量配置远端模型与本地监督服务地址（例如 `DEEPSEEK_API_KEY`、`REMOTE_WORKER_API_BASE`、`LOCAL_LLM_API_BASE` 等）。
