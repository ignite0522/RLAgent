import os
import time
import json
from pathlib import Path
from typing import Any, List, Literal

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ADAPTER_PATH = _PROJECT_ROOT / "runs" / "peft_rl_remote+9714428" / "ep_9"
PEFT_ADAPTER_PATH = os.getenv("PEFT_ADAPTER_PATH", str(_DEFAULT_ADAPTER_PATH)).strip()


def _resolve_base_model_path() -> str:
    """优先使用环境变量；否则尝试从 adapter_config.json 读取基座模型路径。"""
    env_base = os.getenv("BASE_MODEL_PATH", "").strip()
    if env_base:
        return env_base

    if PEFT_ADAPTER_PATH:
        cfg_path = Path(PEFT_ADAPTER_PATH) / "adapter_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                base_from_cfg = str(cfg.get("base_model_name_or_path", "")).strip()
                if base_from_cfg:
                    print(f"[http_server] 从 adapter_config.json 读取基座模型: {base_from_cfg}")
                    return base_from_cfg
            except Exception as e:
                print(f"[http_server] 读取 {cfg_path} 失败: {e}")

    return "/home/user/.cache/modelscope/hub/qwen/Qwen2___5-7B-Instruct"


BASE_MODEL_PATH = _resolve_base_model_path()


if torch.cuda.is_available():
    DEVICE = "cuda"
    TORCH_DTYPE = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    TORCH_DTYPE = torch.float16
else:
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32


print(f"[http_server] 使用设备: {DEVICE}, dtype: {TORCH_DTYPE}")

print("[http_server] 正在加载基座模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=TORCH_DTYPE,
    trust_remote_code=True,
)
if PEFT_ADAPTER_PATH:
    adapter_path = Path(PEFT_ADAPTER_PATH)
    if adapter_path.exists():
        print(f"[http_server] 正在加载 PEFT adapter: {PEFT_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, PEFT_ADAPTER_PATH)
        print(f"[http_server] LoRA 加载成功: {PEFT_ADAPTER_PATH}")
    else:
        print(f"[http_server] 未加载 LoRA：路径不存在 -> {PEFT_ADAPTER_PATH}")
else:
    print("[http_server] 未加载 LoRA：PEFT_ADAPTER_PATH 为空")
model.to(DEVICE)
model.eval()




class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    max_tokens: int = 4096
    temperature: float = 0.7


class OpenAIChatChoiceMessage(BaseModel):
    role: str
    content: str


class OpenAIChatChoice(BaseModel):
    index: int
    message: OpenAIChatChoiceMessage
    finish_reason: str


class OpenAIChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatChoice]
    usage: OpenAIChatUsage


app = FastAPI(title="Qwen OpenAI-Compatible Server", version="0.1")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "device": DEVICE,
        "dtype": str(TORCH_DTYPE),
        "base_model_path": BASE_MODEL_PATH,
    }


@app.post("/v1/chat/completions", response_model=OpenAIChatResponse)
def chat_completions(
    req: OpenAIChatRequest,
) -> OpenAIChatResponse:

    # 使用 Qwen 的对话模板来构造输入
    messages = [m.model_dump() for m in req.messages]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    t0 = int(time.time())
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0,
            temperature=req.temperature if req.temperature > 0 else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 只解码新生成部分
    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    prompt_tokens = int(inputs["input_ids"].numel())
    completion_tokens = int(gen_ids.numel())

    # 成功处理一次请求就打印提示（强制 flush，避免 stdout 缓冲）
    print("请求成功", flush=True)

    return OpenAIChatResponse(
        id="chatcmpl-qwen-local",
        object="chat.completion",
        created=t0,
        model=req.model,
        choices=[
            OpenAIChatChoice(
                index=0,
                message=OpenAIChatChoiceMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=OpenAIChatUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


if __name__ == "__main__":
    import uvicorn

    # 写死监听地址与端口，方便在服务器上直接启动
    # 例如：export OPENAI_API_KEY=xxx && python lora/http_server.py
    uvicorn.run(app, host="0.0.0.0", port=8001)

