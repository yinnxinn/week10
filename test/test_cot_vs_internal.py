# -*- coding: utf-8 -*-
"""
CoT（链式思考）与内生推理的对比测试。
使用硅基流动 API（app.services.llm.LLMService），通过多组例子说明：
- CoT：显式要求「一步步思考/写出推理过程」，模型会输出中间步骤再给答案。
- 内生推理：不要求步骤，模型可能直接给出答案（依赖模型内部的隐式推理）。
"""
import os
import sys
import re

import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 无 API Key 时跳过真实调用（仅做本地 prompt 对比或 mock）
os.environ["SILICONFLOW_API_KEY"] = "sk-yyyvuckwvwpzuanghmegtoszbpezmhfycaihzzsjicidshwc"
os.environ['SILICONFLOW_MODEL'] = 'Qwen/Qwen2.5-14B-Instruct'
SKIP_REAL_LLM = not os.getenv("SILICONFLOW_API_KEY")


def _call_llm(system_prompt: str, user_prompt: str, use_cot: bool):
    """调用硅基流动：与 app.services.llm 一致。"""
    from app.services.llm import LLMService
    llm = LLMService()
    try:
        r = llm.client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.3
        )
        print(r.choices[0].message)
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[API Error] {e}"


# ---------------------------------------------------------------------------
# 例子：题目 + (CoT 系统/用户提示, 内生推理 系统/用户提示)
# ---------------------------------------------------------------------------

EXAMPLES = [
    {
        "name": "逻辑推理",
        "question": "去洗车离我家有20米。我应该开车去还是走路去。",
        "cot_system": "你是一个善于分步推理的助手。请先一步步写出推理过程，再给出最终答案。回答格式：先写「推理过程：」，再写「最终答案：」。",
        "cot_user": "请一步步思考并给出答案。\n\n{question}",
        "internal_system": "你是一个简洁的助手，直接给出答案。",
        "internal_user": "请直接回答，不要写推理过程。\n\n{question}",
    },
    {
        "name": "数学多步运算",
        "question": "小明有 5 个苹果，吃了 2 个，又买了 3 个，现在一共有几个苹果？",
        "cot_system": "你是一个善于分步推理的助手。请先一步步写出推理过程，再给出最终答案。回答格式：先写「推理过程：」，再写「最终答案：」。",
        "cot_user": "请一步步思考并给出答案。\n\n{question}",
        "internal_system": "你是一个简洁的助手，直接给出答案。",
        "internal_user": "请直接回答，不要写推理过程。\n\n{question}",
    },
    {
        "name": "逻辑推理",
        "question": "三人 A、B、C：A 说真话，B 说假话，C 说真话。A 说「B 在说谎」，B 说「C 在说谎」，C 说「A 在说谎」。谁一定在说谎？",
        "cot_system": "你是一个逻辑推理助手。请先一步步分析每个人的陈述是否一致，再给出结论。回答中先写「推理过程：」，再写「最终答案：」。",
        "cot_user": "请一步步分析并给出答案。\n\n{question}",
        "internal_system": "你是一个助手，直接给出结论。",
        "internal_user": "请直接给出谁在说谎，无需写推理。\n\n{question}",
    },
    {
        "name": "简单算术",
        "question": "一个数加上 12 等于 30，这个数是多少？",
        "cot_system": "请先写出推理过程（例如设未知数、列式），再给出最终答案。格式：先「推理过程：」，再「最终答案：」。",
        "cot_user": "请一步步思考并回答。\n\n{question}",
        "internal_system": "直接给出答案即可。",
        "internal_user": "直接回答数字，不要写过程。\n\n{question}",
    },
    {
        "name": "医学建议（需多步推断）",
        "question": "患者发热 38.5℃、咽痛、流涕 2 天，无咳嗽。根据常见病推断，最可能是什么？应该去哪个科室？",
        "cot_system": "你是医疗咨询助手。请先根据症状一步步分析可能病因和科室，再给出建议。回答格式：先写「推理过程：」，再写「最终答案：」。",
        "cot_user": "请一步步分析症状并给出就诊建议。\n\n{question}",
        "internal_system": "你是医疗助手，直接给出结论和建议。",
        "internal_user": "请直接给出最可能的诊断和科室建议，无需写分析过程。\n\n{question}",
    },
    {
        "name": "常识与多步推理",
        "question": "一个班级 40 人，其中 60% 是女生。女生比男生多几人？",
        "cot_system": "请先写出推理过程（先算女生人数、男生人数，再算差值），再给出最终答案。格式：先「推理过程：」，再「最终答案：」。",
        "cot_user": "请一步步计算并回答。\n\n{question}",
        "internal_system": "直接给出最终数字答案。",
        "internal_user": "直接回答女生比男生多几人，只给数字即可。\n\n{question}",
    },
]


def _has_explicit_reasoning(text: str) -> bool:
    """粗略判断回复中是否包含显式推理表述。"""
    if not text:
        return False
    markers = ["推理过程", "第一步", "步骤", "首先", "其次", "因此", "因为", "所以", "分析：", "计算"]
    return any(m in text for m in markers)


def _has_final_answer_marker(text: str) -> bool:
    """是否包含「最终答案」等收束标记。"""
    return "最终答案" in text or "答案：" in text


@pytest.mark.skipif(SKIP_REAL_LLM, reason="需要 SILICONFLOW_API_KEY 才调用硅基流动")
class TestCotVsInternalRealAPI:
    """CoT vs 内生推理：真实调用硅基流动 API，多例子对比。"""

    @pytest.mark.parametrize("example", EXAMPLES, ids=[e["name"] for e in EXAMPLES])
    def test_cot_response_contains_reasoning_markers(self, example):
        """CoT 条件下，模型回复应包含推理过程标记或步骤表述。"""
        q = example["question"]
        user = example["cot_user"].format(question=q)
        content = _call_llm(example["cot_system"], user, use_cot=True)
        assert content, "CoT 回复不应为空"
        assert _has_explicit_reasoning(content) or _has_final_answer_marker(content), (
            f"CoT 回复应包含推理表述或「最终答案」: {content[:200]}..."
        )

    @pytest.mark.parametrize("example", EXAMPLES, ids=[e["name"] for e in EXAMPLES])
    def test_internal_response_often_shorter_or_direct(self, example):
        """内生推理条件下，回复往往更短或更直接（不强制要求步骤）。"""
        q = example["question"]
        user = example["internal_user"].format(question=q)
        content = _call_llm(example["internal_system"], user, use_cot=False)
        assert content, "内生推理回复不应为空"
        # 内生推理不要求写步骤，长度通常 ≤ CoT（仅作软性观察，不断言）
        assert len(content) >= 1

    def test_cot_vs_internal_side_by_side(self):
        """选一题同时跑 CoT 与内生推理，对比长度与是否含推理标记。"""
        example = EXAMPLES[0]
        q = example["question"]

        cot_user = example["cot_user"].format(question=q)
        cot_content = _call_llm(example["cot_system"], cot_user, use_cot=True)

        internal_user = example["internal_user"].format(question=q)
        internal_content = _call_llm(example["internal_system"], internal_user, use_cot=False)

        assert cot_content, "CoT 回复不应为空"
        assert internal_content, "内生推理回复不应为空"

        has_cot_reasoning = _has_explicit_reasoning(cot_content) or _has_final_answer_marker(cot_content)
        has_internal_reasoning = _has_explicit_reasoning(internal_content)

        # CoT 应更常出现显式推理
        assert has_cot_reasoning, f"CoT 应有推理标记: {cot_content[:300]}"
        # 内生推理可能没有（取决于模型），这里只打印便于人工对比
        print("\n--- CoT ---\n" + cot_content[:500])
        print("\n--- 内生推理 ---\n" + internal_content[:500])


class TestCotVsInternalPromptOnly:
    """不调 API，只校验 prompt 构建与解析逻辑（CoT vs 内生 的 prompt 差异）。"""

    def test_cot_prompt_asks_for_steps(self):
        for ex in EXAMPLES:
            user = ex["cot_user"].format(question=ex["question"])
            assert "一步步" in user or "推理" in ex["cot_system"], ex["name"]

    def test_internal_prompt_asks_direct_answer(self):
        for ex in EXAMPLES:
            user = ex["internal_user"].format(question=ex["question"])
            assert "直接" in user or "无需" in user or "不要" in user, ex["name"]

    def test_has_explicit_reasoning_detection(self):
        assert _has_explicit_reasoning("推理过程：首先… 最终答案：6")
        assert _has_explicit_reasoning("第一步：列式 第二步：计算")
        assert not _has_explicit_reasoning("")

    def test_final_answer_marker_detection(self):
        assert _has_final_answer_marker("最终答案：6 个")
        assert _has_final_answer_marker("答案：呼吸内科")
        assert not _has_final_answer_marker("仅有一段话无标记")


def run_examples_standalone():
    """直接运行脚本时，对每个例子打印 CoT vs 内生推理 的对比（需配置 API Key）。"""
    if SKIP_REAL_LLM:
        print("未设置 SILICONFLOW_API_KEY，跳过真实调用。仅做 prompt 预览。")
        for ex in EXAMPLES:
            q = ex["question"]
            print(f"\n【{ex['name']}】")
            print("题目:", q)
            print("CoT user:", ex["cot_user"].format(question=q)[:120], "...")
            print("内生 user:", ex["internal_user"].format(question=q)[:120], "...")
        return

    for ex in EXAMPLES:
        q = ex["question"]
        print("\n" + "=" * 60)
        print(f"【{ex['name']}】 {q}")
        print("=" * 60)
        cot_user = ex["cot_user"].format(question=q)
        internal_user = ex["internal_user"].format(question=q)
        try:
            cot_r = _call_llm(ex["cot_system"], cot_user, use_cot=True)
            print("\n--- CoT 回复 ---\n" + cot_r + "\n")
            internal_r = _call_llm(ex["internal_system"], internal_user, use_cot=False)
            print("--- 内生推理 回复 ---\n" + internal_r + "\n")
        except Exception as e:
            print(f"调用失败: {e}")


if __name__ == "__main__":
    run_examples_standalone()
