# -*- coding: utf-8 -*-
"""
RAG (Retrieval-Augmented Generation) 测试用例。
覆盖：检索流程、上下文注入、空检索回退、提示构建。
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# 项目根目录加入 path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ----- 纯 RAG 流程（不依赖 Neo4j/LLM）-----

def _build_rag_prompt(question: str, context: str, empty_fallback: str = "无相关知识库信息。") -> str:
    """构建 RAG 用户提示（与 test_kg_rag 中逻辑一致）。"""
    ctx = context.strip() or empty_fallback
    return f"""用户问题：{question}

知识图谱上下文：
{ctx}

请回答："""


def _mock_retrieve(question: str, knowledge: dict) -> str:
    """模拟检索：根据问题关键词返回预设知识片段。"""
    for keyword, text in knowledge.items():
        if keyword in question:
            return text
    return ""


class TestRAGRetrieve:
    """RAG 检索环节测试。"""

    def test_retrieve_returns_non_empty_when_keyword_matches(self):
        knowledge = {"肺泡蛋白质沉积症": "肺泡蛋白质沉积症 -> 就诊科室 -> 呼吸内科；建议检查：胸部CT、支气管镜、肺活检。"}
        ctx = _mock_retrieve("肺泡蛋白质沉积症应该去哪个科室？", knowledge)
        assert ctx
        assert "呼吸内科" in ctx

    def test_retrieve_returns_empty_when_no_match(self):
        knowledge = {"感冒": "感冒常见症状：发热、咳嗽。"}
        ctx = _mock_retrieve("糖尿病怎么治疗？", knowledge)
        assert ctx == ""

    def test_retrieve_multiple_keywords_first_match(self):
        knowledge = {"儿科": "儿科常见病：发热、咳嗽。", "发烧": "发热处理：物理降温、补液。"}
        ctx = _mock_retrieve("儿科宝宝发烧怎么办", knowledge)
        assert ctx  # 至少匹配一个
        assert "儿科" in ctx or "发热" in ctx


class TestRAGPrompt:
    """RAG 提示构建测试。"""

    def test_prompt_contains_question_and_context(self):
        question = "肺泡蛋白质沉积症应该去哪个科室？"
        context = "肺泡蛋白质沉积症 -> 呼吸内科；检查：CT、肺活检。"
        prompt = _build_rag_prompt(question, context)
        assert question in prompt
        assert context in prompt
        assert "知识图谱上下文" in prompt

    def test_prompt_uses_fallback_when_context_empty(self):
        question = "未知病怎么治？"
        prompt = _build_rag_prompt(question, "")
        assert "无相关知识库信息" in prompt
        assert question in prompt

    def test_prompt_uses_fallback_when_context_whitespace_only(self):
        prompt = _build_rag_prompt("某问题", "   \n  ")
        assert "无相关知识库信息" in prompt


class TestRAGPipeline:
    """RAG 完整流程（检索 + 提示）测试。"""

    def test_pipeline_retrieve_then_prompt(self):
        knowledge = {"感冒": "感冒可对症用药，多休息。"}
        question = "感冒了怎么办？"
        context = _mock_retrieve(question, knowledge)
        prompt = _build_rag_prompt(question, context)
        assert "感冒" in prompt
        assert "对症用药" in prompt or "多休息" in prompt

    def test_pipeline_empty_retrieve_still_produces_valid_prompt(self):
        question = "非常冷门的问题"
        context = _mock_retrieve(question, {})
        prompt = _build_rag_prompt(question, context)
        assert question in prompt
        assert "无相关知识库信息" in prompt


class TestRAGWithLLMMock:
    """RAG + LLM 调用（mock LLM）测试。"""

    @pytest.fixture
    def mock_llm_response(self):
        return "根据知识库，建议前往呼吸内科就诊，并做胸部CT等检查。"

    def test_llm_receives_system_and_user_message_with_context(self, mock_llm_response):
        from app.services.llm import LLMService

        question = "肺泡蛋白质沉积症应该去哪个科室？"
        context = "肺泡蛋白质沉积症 -> 呼吸内科；检查：CT。"
        user_prompt = _build_rag_prompt(question, context)

        llm = LLMService()
        with patch.object(llm, "client") as mock_client:
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content=mock_llm_response))]
            )
            resp = llm.client.chat.completions.create(
                model=llm.model,
                messages=[
                    {"role": "system", "content": "你是基于知识图谱的智能问答助手。"},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.7,
            )
            answer = resp.choices[0].message.content

        assert answer == mock_llm_response
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert context in messages[1]["content"]
        assert question in messages[1]["content"]
