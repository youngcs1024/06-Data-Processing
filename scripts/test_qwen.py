from llm.qwen_client import chat_once
from secrets_local import DASHSCOPE_API_KEY

print(
    chat_once(
        api_key=DASHSCOPE_API_KEY,
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "用一句话自我介绍。"},
        ],
    )
)
