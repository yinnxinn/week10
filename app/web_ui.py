import streamlit as st
import sys
import os
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from app.services.knowledge_base import KnowledgeBase
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.core.config import settings

st.set_page_config(page_title="多模态医疗咨询助手", page_icon="🏥", layout="wide")

@st.cache_resource
def load_services():
    with st.spinner("正在加载模型和服务..."):
        embedding_service = EmbeddingService("openai/clip-vit-base-patch32")
        kb = KnowledgeBase()
        llm = LLMService()
        return embedding_service, kb, llm

try:
    embedding_service, kb, llm = load_services()
except Exception as e:
    st.error(f"服务加载失败: {e}")
    st.stop()

st.title("🏥 多模态医疗咨询助手")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for settings or info
with st.sidebar:
    st.header("系统状态")
    st.success("服务已连接")
    st.info("本系统支持文本和图像多模态检索。")
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

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
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("正在分析医疗数据..."):
            try:
                # 1. Embed
                query_vector = embedding_service.embed_query(prompt)[0]
                
                # 2. Text Search (Hybrid)
                text_results = kb.hybrid_search(prompt, query_vector, top_k=3, rerank=True)
                
                # 3. Image Search (Text-to-Image)
                image_results = kb.search_images(query_vector, top_k=3)
                
                # Logic Check for Department
                top_dept = None
                
                # Check top result's department
                if text_results and text_results[0]:
                    top_hit = text_results[0][0]
                    entity = top_hit.get("entity", {})
                    dept = entity.get("department", "")
                    # Filter out invalid departments
                    if dept and str(dept).lower() not in ["unknown", "nan", "none", ""]:
                        top_dept = dept
                
                display_images = []
                response_content = ""
                
                if top_dept:
                    # Case 1: Department Found -> Recommend it
                    response_content += f"### ✅ 建议就诊科室：{top_dept}\n\n"
                    response_content += "根据您的描述，匹配到相关病例属于该科室。建议您前往该科室进行详细检查及咨询。\n\n"
                    
                    # LLM Analysis for more details
                    context = ""
                    for hits in text_results:
                        for hit in hits:
                            entity = hit.get('entity', {})
                            context += f"病例: {entity.get('text', '')}\n科室: {entity.get('department', '')}\n\n"
                    
                    suggestion = llm.get_consultation_suggestion(prompt, context)
                    response_content += f"#### 📋 智能分析：\n{suggestion}"
                    
                else:
                    # Case 2: No Department -> Show Images + Text + Suggestion
                    response_content += "未匹配到明确的建议科室。以下是为您找到的相关参考资料和建议：\n\n"
                    
                    # Collect Images
                    if image_results and image_results[0]:
                        for hit in image_results[0]:
                            entity = hit.get('entity', {})
                            img_path = entity.get('image_path')
                            if img_path and os.path.exists(img_path):
                                if img_path not in display_images:
                                    display_images.append(img_path)
                    
                    # Context for LLM
                    context = "文本资料:\n"
                    if text_results and text_results[0]:
                        for hit in text_results[0]:
                            context += f"- {hit['entity'].get('text', '')}\n"
                    
                    context += "\n影像资料:\n"
                    if image_results and image_results[0]:
                        for hit in image_results[0]:
                            caption = hit['entity'].get('text', '')
                            context += f"- {caption}\n"

                    # LLM Suggestion
                    suggestion = llm.get_consultation_suggestion(prompt, context)
                    response_content += f"#### 🩺 诊疗建议：\n{suggestion}"
                
                # Display Response
                message_placeholder.markdown(response_content)
                
                # Display Images (if any, primarily for Case 2 but good to show if available)
                if display_images:
                    st.write("#### 🖼️ 参考影像：")
                    cols = st.columns(len(display_images))
                    for idx, img_path in enumerate(display_images):
                        with cols[idx % 3]:
                            st.image(img_path, caption="检索结果", use_column_width=True)
                
                # Save to history
                msg_entry = {"role": "assistant", "content": response_content}
                if display_images:
                    msg_entry["images"] = display_images
                st.session_state.messages.append(msg_entry)
                
            except Exception as e:
                st.error(f"处理请求时发生错误: {e}")
