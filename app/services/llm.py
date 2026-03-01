from openai import OpenAI
import os
import re
from typing import Iterator

class LLMService:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY") or "sk-yyyvuckwvwpzuanghmegtoszbpezmhfycaihzzsjicidshwc"
        self.base_url = base_url or os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1"
        self.model = model or os.getenv("SILICONFLOW_MODEL") or "Qwen/Qwen2.5-14B-Instruct"#"Pro/zai-org/GLM-4.7"
        
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

    def get_consultation_suggestion_stream(self, query: str, context: str) -> Iterator[str]:
        """
        基于用户问题和检索到的上下文流式生成就诊建议，逐 chunk 产出内容。
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
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1024,
                temperature=0.7,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield delta.content
        except Exception as e:
            print(f"LLM Error: {e}")
            yield "抱歉，由于网络或服务原因，暂时无法生成智能建议。请您根据上述检索结果参考，或直接前往医院咨询导诊台。"

    # ---------- 推理过程：模型 Think 输出 ----------
    THINK_SYSTEM = """你是一个专业的医疗咨询助手。请按以下格式回答：
1. 先将你的内心推理过程写在 <think>...</think> 标签内（分析用户症状、参考信息、可能病因与科室）。
2. 然后在「最终回答：」后给出给用户的就诊建议。
务必包含 <think> 和 </think> 标签。"""

    def get_think_output(self, query: str, context: str) -> str:
        """获取模型「内心推理」输出，要求模型将思考写在 <think>...</think> 中。返回完整回复，由调用方解析。"""
        user_prompt = f"""用户问题：{query}

参考信息：
{context}

请先将内心推理写在 <think></think> 中，再在「最终回答：」后给出建议。"""
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.THINK_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048,
                temperature=0.3,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"LLM Error (think): {e}")
            return ""

    @staticmethod
    def parse_think_response(text: str) -> tuple[str, str]:
        """从回复中解析 <think>...</think> 与「最终回答：」后的内容。返回 (think_content, final_answer)。"""
        think, final = "", ""
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m:
            think = m.group(1).strip()
        if "最终回答：" in text:
            final = text.split("最终回答：", 1)[1].strip()
        elif "最终答案：" in text:
            final = text.split("最终答案：", 1)[1].strip()
        return think, final

    # ---------- 推理过程：CoT 输出 ----------
    COT_SYSTEM = """你是一个善于分步推理的医疗咨询助手。请根据用户问题和参考信息，按以下格式回答：
1. 先写「推理过程：」，然后一步步分析症状、参考信息、可能病因与科室建议。
2. 再写「最终答案：」，给出简洁的就诊建议总结。
务必包含「推理过程：」和「最终答案：」两段。"""

    def get_cot_output_stream(self, query: str, context: str) -> Iterator[str]:
        """CoT 格式流式输出（推理过程 + 最终答案），逐 chunk 产出。"""
        user_prompt = f"""用户问题：{query}

参考信息：
{context}

请先写「推理过程：」一步步分析，再写「最终答案：」给出建议。"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.COT_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield delta.content
        except Exception as e:
            print(f"LLM Error (cot): {e}")
            yield ""

    @staticmethod
    def parse_cot_response(text: str) -> tuple[str, str]:
        """从 CoT 回复中解析「推理过程：」与「最终答案：」。返回 (reasoning, final_answer)。"""
        reasoning, final = "", ""
        if "推理过程：" in text:
            parts = text.split("推理过程：", 1)[1]
            if "最终答案：" in parts:
                reasoning = parts.split("最终答案：", 1)[0].strip()
                final = parts.split("最终答案：", 1)[1].strip()
            else:
                reasoning = parts.strip()
        elif "最终答案：" in text:
            final = text.split("最终答案：", 1)[1].strip()
        else:
            final = text.strip()
        return reasoning, final
