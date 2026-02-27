# -*- coding: utf-8 -*-
"""
CoT (Chain of Thought) 测试用例。
覆盖：分步推理提示构建、步骤抽取、答案与推理分离。
"""
import os
import sys
import re
from unittest.mock import MagicMock, patch

import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ----- CoT 提示与解析 -----

COT_SYSTEM_PROMPT = """你是一个善于分步推理的助手。请先一步步分析问题，再给出最终答案。
要求：在回答中先写「推理过程：」，再写「最终答案：」。"""


def build_cot_user_prompt(question: str, cot_trigger: str = "请一步步思考并给出答案。") -> str:
    """构建 CoT 风格的用户提示。"""
    return f"{question}\n\n{cot_trigger}"


def parse_cot_response(response: str) -> tuple[str, str]:
    """
    从模型回复中解析「推理过程」和「最终答案」。
    返回 (reasoning, answer)，若无法解析则 answer 为整段回复。
    """
    reasoning = ""
    answer = response.strip()

    if "推理过程：" in response:
        parts = response.split("推理过程：", 1)
        if "最终答案：" in parts[1]:
            mid, ans = parts[1].split("最终答案：", 1)
            reasoning = mid.strip()
            answer = ans.strip()
        else:
            reasoning = parts[1].strip()
    elif "最终答案：" in response:
        _, answer = response.split("最终答案：", 1)
        answer = answer.strip()

    return reasoning, answer


def extract_steps(text: str) -> list[str]:
    """从推理文本中抽取步骤（按 步骤1/第一步/1. 等模式）。"""
    steps = []
    # 匹配 "步骤1" "第一步" "1." "1、" 等
    pattern = r"(?:步骤\s*)?(?:第)?[一二三四五六七八九十\d]+[.、步]\s*[^\n]+"
    for m in re.finditer(pattern, text):
        steps.append(m.group(0).strip())
    if not steps:
        # 按换行分句作为备选
        steps = [s.strip() for s in text.split("\n") if s.strip()]
    return steps


class TestCOTPrompt:
    """CoT 提示构建测试。"""

    def test_cot_prompt_contains_question_and_trigger(self):
        q = "小明有5个苹果，吃了2个，又买了3个，现在有几个？"
        prompt = build_cot_user_prompt(q)
        assert q in prompt
        assert "一步步思考" in prompt

    def test_cot_prompt_custom_trigger(self):
        q = "某医学问题"
        trigger = "请先分析病因，再给出诊疗建议。"
        prompt = build_cot_user_prompt(q, cot_trigger=trigger)
        assert trigger in prompt
        assert "病因" in prompt


class TestCOTParse:
    """CoT 回复解析测试。"""

    def test_parse_reasoning_and_answer(self):
        response = """推理过程：
首先确定已知：5个，吃了2个，剩3个；又买3个，共6个。
最终答案：6个苹果。"""
        reasoning, answer = parse_cot_response(response)
        assert "首先确定" in reasoning or "已知" in reasoning
        assert "6" in answer

    def test_parse_only_final_answer(self):
        response = "最终答案：建议前往呼吸内科。"
        reasoning, answer = parse_cot_response(response)
        assert "呼吸内科" in answer
        assert reasoning == ""

    def test_parse_no_markers_returns_full_as_answer(self):
        response = "直接回复：多喝水多休息。"
        reasoning, answer = parse_cot_response(response)
        assert answer == response.strip()
        assert reasoning == ""

    def test_extract_steps_from_reasoning(self):
        text = "步骤1：先算剩余 5-2=3。步骤2：再算总共 3+3=6。"
        steps = extract_steps(text)
        assert len(steps) >= 1
        assert any("5-2" in s or "3+3" in s for s in steps)


class TestCOTWithLLMMock:
    """CoT + LLM 调用（mock）测试。"""

    def test_llm_receives_cot_style_messages(self):
        from app.services.llm import LLMService

        question = "10支笔分给2人，每人几支？"
        user_prompt = build_cot_user_prompt(question)
        mock_content = "推理过程：10÷2=5。最终答案：每人5支。"

        llm = LLMService()
        with patch.object(llm, "client") as mock_client:
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content=mock_content))]
            )
            resp = llm.client.chat.completions.create(
                model=llm.model,
                messages=[
                    {"role": "system", "content": COT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            content = resp.choices[0].message.content

        reasoning, answer = parse_cot_response(content)
        assert "5" in answer
        assert "10" in content or "2" in content
