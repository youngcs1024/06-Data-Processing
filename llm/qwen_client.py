from __future__ import annotations

from openai import OpenAI

# 北京 base_url（阿里云文档）:contentReference[oaicite:7]{index=7}
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

def make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=BASE_URL)

def chat_once(*, api_key: str, model: str, messages: list[dict], temperature: float = 0.2) -> str:
    client = make_client(api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content
