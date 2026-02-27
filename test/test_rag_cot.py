# -*- coding: utf-8 -*-
"""
RAG + CoT 联合测试用例。
覆盖：先检索再分步推理、上下文与 CoT 提示组合、端到端流程（mock）。
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ----- 本地复用：RAG / CoT 的构建与解析（避免跨 test 模块导入）-----

def _build_rag_prompt(question: str, context: str, empty_fallback: str = "无相关知识库信息。") -> str:
    ctx = context.strip() or empty_fallback
    return f"""用户问题：{question}

知识图谱上下文：
{ctx}

请回答："""


def _mock_retrieve(question: str, knowledge: dict) -> str:
    for keyword, text in knowledge.items():
        if keyword in question:
            return text
    return ""


def build_cot_user_prompt(question: str, cot_trigger: str = "请一步步思考并给出答案。") -> str:
    return f"{question}\n\n{cot_trigger}"


def parse_cot_response(response: str) -> tuple:
    reasoning, answer = "", response.strip()
    if "推理过程：" in response:
        parts = response.split("推理过程：", 1)
        if "最终答案：" in parts[1]:
            mid, ans = parts[1].split("最终答案：", 1)
            reasoning, answer = mid.strip(), ans.strip()
        else:
            reasoning = parts[1].strip()
    elif "最终答案：" in response:
        _, answer = response.split("最终答案：", 1)
        answer = answer.strip()
    return reasoning, answer


RAG_COT_SYSTEM = """你是一个基于知识库且善于分步推理的医疗助手。
请结合提供的[知识库上下文]先一步步分析问题，再给出最终建议。
回答格式：先写「推理过程：」，再写「最终答案：」。"""


def build_rag_cot_prompt(question: str, context: str, empty_fallback: str = "无相关知识库信息。") -> str:
    """RAG 上下文 + CoT 要求的用户提示。"""
    ctx = context.strip() or empty_fallback
    return f"""用户问题：{question}

知识库上下文：
{ctx}

请先一步步分析上述信息与问题的关系，再给出最终答案。"""


def rag_cot_flow(question: str, retrieve_fn, llm_create_fn):
    """
    执行 RAG+CoT 流程：检索 -> 构建 RAG+CoT 提示 -> 调用 LLM。
    retrieve_fn(question) -> context; llm_create_fn(messages) -> response content.
    """
    context = retrieve_fn(question)
    user_prompt = build_rag_cot_prompt(question, context)
    messages = [
        {"role": "system", "content": RAG_COT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    return llm_create_fn(messages)


class TestRAGCoTPrompt:
    """RAG+CoT 提示构建测试。"""

    def test_prompt_contains_rag_context_and_cot_instruction(self):
        question = "肺泡蛋白质沉积症应该去哪个科室？"
        context = "肺泡蛋白质沉积症 -> 呼吸内科；检查：CT、肺活检。"
        prompt = build_rag_cot_prompt(question, context)
        assert question in prompt
        assert context in prompt
        assert "知识库上下文" in prompt
        assert "一步步分析" in prompt or "最终答案" in prompt

    def test_prompt_empty_context_uses_fallback(self):
        prompt = build_rag_cot_prompt("某问题", "")
        assert "无相关知识库信息" in prompt


class TestRAGCoTPipeline:
    """RAG+CoT 流程测试（mock 检索与 LLM）。"""

    def test_flow_retrieve_then_prompt_includes_context(self):
        knowledge = {"感冒": "感冒可对症用药，多休息，必要时发热用药。"}
        question = "感冒了怎么办？"
        context = _mock_retrieve(question, knowledge)
        prompt = build_rag_cot_prompt(question, context)
        assert "感冒" in prompt
        assert "对症用药" in prompt or "多休息" in prompt
        assert "一步步" in prompt

    def test_flow_llm_receives_rag_context_and_cot_instruction(self):
        question = "某病去哪个科室？"
        context = "某病 -> 内科。"
        mock_content = "推理过程：根据知识库，某病对应内科。最终答案：建议挂内科。"

        def fake_retrieve(q):
            return context

        def fake_llm(messages):
            return mock_content

        result = rag_cot_flow(question, fake_retrieve, fake_llm)
        assert result == mock_content

        full_prompt = build_rag_cot_prompt(question, context)
        assert context in full_prompt
        assert question in full_prompt

    def test_flow_parsed_reasoning_and_answer(self):
        mock_content = "推理过程：根据知识库该病属呼吸系统。最终答案：建议呼吸内科就诊。"
        reasoning, answer = parse_cot_response(mock_content)
        assert "呼吸" in reasoning or "呼吸" in answer
        assert "呼吸内科" in answer


class TestRAGCoTWithLLMMock:
    """RAG+CoT 与 LLM 集成（mock LLM）测试。"""

    def test_llm_called_with_rag_context_and_cot_system_prompt(self):
        from app.services.llm import LLMService

        question = "小儿发烧怎么办？"
        context = "发热处理：物理降温、补液；高热需就医。"
        user_prompt = build_rag_cot_prompt(question, context)
        mock_content = "推理过程：结合知识库，先物理降温。最终答案：可先物理降温，高热及时就医。"

        llm = LLMService()
        with patch.object(llm, "client") as mock_client:
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content=mock_content))]
            )
            resp = llm.client.chat.completions.create(
                model=llm.model,
                messages=[
                    {"role": "system", "content": RAG_COT_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.5,
            )
            content = resp.choices[0].message.content

        assert "推理过程" in content or "最终答案" in content
        reasoning, answer = parse_cot_response(content)
        assert "物理降温" in content
        call_args = mock_client.chat.completions.create.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        assert context in user_content
        assert question in user_content
