import streamlit as st
import sys
import os
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

st.set_page_config(page_title="多模态医疗咨询助手", page_icon="🏥", layout="wide")

# 先渲染标题和侧栏，避免因 load_services 阻塞导致白屏
st.title("🏥 多模态医疗咨询助手")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("系统状态")
    st.info("本系统支持文本和图像多模态检索。")
    st.caption("服务将在您首次发送消息时加载，请稍候。")
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

# 延迟加载：仅在用户首次提交时加载，保证首屏立即显示
@st.cache_resource
def load_services():
    from app.services.knowledge_base import KnowledgeBase
    from app.services.embeddings import EmbeddingService
    from app.services.llm import LLMService
    embedding_service = EmbeddingService("openai/clip-vit-base-patch32")
    kb = KnowledgeBase()
    llm = LLMService()
    return embedding_service, kb, llm

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            cols = st.columns(min(len(message["images"]), 3))
            for idx, img_path in enumerate(message["images"]):
                if idx < 3: # Limit display in history to avoid clutter
                    with cols[idx]:
                        if os.path.exists(img_path):
                            st.image(img_path, caption="参考影像", use_column_width=True)

prompt = st.chat_input("请描述您的症状或上传医学影像...")

if prompt:
    # 首次提交时加载服务（可能阻塞），失败则提示并停止
    try:
        embedding_service, kb, llm = load_services()
    except Exception as e:
        st.error(f"服务加载失败: {e}")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner("正在检索知识库..."):
                # 1. Embed
                ## 优化点：
                ## 1. 根据输入类型判断，支持图像输入，文件上传
                ## 2. 多链路检索，结果合并
                query_vector = embedding_service.embed_query(prompt)[0]
                # 2. Text Search (Hybrid)
                text_results = kb.hybrid_search(prompt, query_vector, top_k=3, rerank=True)
                # 3. Image Search (Text-to-Image)
                image_results = kb.search_images(query_vector, top_k=3)

            # Logic Check for Department
            top_dept = None
            if text_results and text_results[0]:
                top_hit = text_results[0][0]
                entity = top_hit.get("entity", {})
                dept = entity.get("department", "")
                if dept and str(dept).lower() not in ["unknown", "nan", "none", ""]:
                    top_dept = dept

            display_images = []
            response_content = ""

            if top_dept:
                response_content += f"### ✅ 建议就诊科室：{top_dept}\n\n"
                response_content += "根据您的描述，匹配到相关病例属于该科室。建议您前往该科室进行详细检查及咨询。\n\n"

                context = ""
                for hits in text_results:
                    for hit in hits:
                        entity = hit.get('entity', {})
                        context += f"病例: {entity.get('text', '')}\n科室: {entity.get('department', '')}\n\n"

                response_content += '---\n'
                response_content += context
                response_content += '---'

                # 推理过程：先展示模型 Think，再展示 CoT
                message_placeholder.markdown(
                    response_content + "\n#### 🧠 推理过程\n"
                )
                full_response = response_content + "\n#### 🧠 推理过程\n"

                # 1) 模型思考 (think)
                think_raw = llm.get_think_output(prompt, context)
                think_text, _ = llm.parse_think_response(think_raw)
                if think_text:
                    full_response += "\n**模型思考 (think)**\n\n" + think_text + "\n\n"
                else:
                    full_response += "\n**模型思考 (think)**\n\n" + (think_raw[:500] if think_raw else "（未解析到 <think> 内容）") + "\n\n"
                message_placeholder.markdown(full_response)

                # 2) CoT 推理（流式），并解析出最终答案
                full_response += "**CoT 推理**\n\n"
                message_placeholder.markdown(full_response)
                cot_streamed = ""
                for chunk in llm.get_cot_output_stream(prompt, context):
                    cot_streamed += chunk
                    # 实时展示 CoT 全文（推理过程+最终答案）
                    full_response_temp = full_response + cot_streamed
                    message_placeholder.markdown(full_response_temp)
                cot_reasoning, cot_final = llm.parse_cot_response(cot_streamed)
                full_response = full_response + cot_streamed

                # 3) 最终结果（使用 CoT 的最终答案）
                full_response += "\n\n#### 📋 最终结果\n\n" + (cot_final if cot_final else cot_streamed)
                message_placeholder.markdown(full_response)
                response_content = full_response

            else:
                response_content += "未匹配到明确的建议科室。以下是为您找到的相关参考资料和建议：\n\n"

                if image_results and image_results[0]:
                    for hit in image_results[0]:
                        entity = hit.get('entity', {})
                        img_path = entity.get('image_path')
                        if img_path and os.path.exists(img_path) and img_path not in display_images:
                            display_images.append(img_path)

                context = "文本资料:\n"
                if text_results and text_results[0]:
                    for hit in text_results[0]:
                        context += f"- {hit['entity'].get('text', '')}\n"
                context += "\n影像资料:\n"
                if image_results and image_results[0]:
                    for hit in image_results[0]:
                        context += f"- {hit['entity'].get('text', '')}\n"

                # 推理过程：模型 Think + CoT
                message_placeholder.markdown(
                    response_content + "\n#### 🧠 推理过程\n"
                )
                full_response = response_content + "\n#### 🧠 推理过程\n"

                # 1) 模型思考 (think)
                think_raw = llm.get_think_output(prompt, context)
                think_text, _ = llm.parse_think_response(think_raw)
                if think_text:
                    full_response += "\n**模型思考 (think)**\n\n" + think_text + "\n\n"
                else:
                    full_response += "\n**模型思考 (think)**\n\n" + (think_raw[:500] if think_raw else "（未解析到 <think> 内容）") + "\n\n"
                message_placeholder.markdown(full_response)

                # 2) CoT 推理（流式）
                full_response += "**CoT 推理**\n\n"
                message_placeholder.markdown(full_response)
                cot_streamed = ""
                for chunk in llm.get_cot_output_stream(prompt, context):
                    cot_streamed += chunk
                    full_response_temp = full_response + cot_streamed
                    message_placeholder.markdown(full_response_temp)
                cot_reasoning, cot_final = llm.parse_cot_response(cot_streamed)
                full_response = full_response + cot_streamed

                # 3) 最终结果
                full_response += "\n\n#### 🩺 最终结果（诊疗建议）\n\n" + (cot_final if cot_final else cot_streamed)
                message_placeholder.markdown(full_response)
                response_content = full_response

            # 参考影像（若有）
            if display_images:
                st.write("#### 🖼️ 参考影像：")
                cols = st.columns(min(len(display_images), 3))
                for idx, img_path in enumerate(display_images):
                    with cols[idx % 3]:
                        if os.path.exists(img_path):
                            st.image(img_path, caption="检索结果", use_column_width=True)

            msg_entry = {"role": "assistant", "content": response_content}
            if display_images:
                msg_entry["images"] = display_images
            st.session_state.messages.append(msg_entry)

        except Exception as e:
            st.error(f"处理请求时发生错误: {e}")

## gradio / streamlit
### python -m streamlit run app/web_ui.py

## 模型流式输出： websocket
## SSE（Server-Sent Events）