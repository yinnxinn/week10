import os
try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' module not found. Please run 'pip install openai'")
    exit(1)

# 配置参数
API_KEY = "sk-lyrzoilwprfhhpeddrmoqzbvwlxnqzezjphlqqsvzxjbebra"  # 默认占位符
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "Pro/zai-org/GLM-4.7"

# 尝试从环境变量读取 API Key（更安全的做法）
env_key = os.getenv("SILICONFLOW_API_KEY")
if env_key:
    API_KEY = env_key
    print("Using API Key from environment variable 'SILICONFLOW_API_KEY'")
else:
    print(f"Using hardcoded API Key: {API_KEY}")

print("-" * 50)
print(f"Testing LLM Connection...")
print(f"Base URL: {BASE_URL}")
print(f"Model:    {MODEL}")
print("-" * 50)

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": "你好，请回复'Test OK'"}
        ],
        max_tokens=20
    )
    print("\n[Success] Response received:")
    print(response.choices[0].message.content)
except Exception as e:
    print("\n[Failed] An error occurred:")
    print(e)
    
    # 简单的故障诊断建议
    error_str = str(e)
    if "401" in error_str:
        print("\nPossible Cause: Invalid API Key. Please update 'API_KEY' in the script or set 'SILICONFLOW_API_KEY' environment variable.")
    elif "404" in error_str:
        print("\nPossible Cause: Model not found or Endpoint URL is incorrect.")
    elif "Connection" in error_str:
        print("\nPossible Cause: Network connection issue or DNS resolution failure.")
