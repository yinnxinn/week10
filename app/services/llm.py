from openai import OpenAI
import os

class LLMService:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY") or "sk-lyrzoilwprfhhpeddrmoqzbvwlxnqzezjphlqqsvzxjbebra"
        self.base_url = base_url or os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1"
        self.model = model or os.getenv("SILICONFLOW_MODEL") or "Pro/zai-org/GLM-4.7"
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def get_consultation_suggestion(self, query: str, context: str) -> str:
        """
        基于用户问题和检索到的上下文生成就诊建议
        """
        system_prompt = """你是一个专业的医疗咨询助手。请根据用户的问题和提供的参考信息（包括相似病例、医学知识等），给出合理的就诊建议。
如果参考信息中包含明确的科室建议，请优先参考。
如果参考信息不足，请根据你的医学知识给出一般性的建议，但要提醒用户仅供参考，及时就医。
请保持语气亲切、专业。"""

        user_prompt = f"""用户问题：{query}

参考信息：
{context}

请给出就诊建议："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1024,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return "抱歉，由于网络或服务原因，暂时无法生成智能建议。请您根据上述检索结果参考，或直接前往医院咨询导诊台。"
