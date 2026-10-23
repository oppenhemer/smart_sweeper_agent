"""
Streamlit 前端入口
流程：页面配置 → 会话初始化 → 侧边栏管理 → 历史渲染 → 用户输入 → 流式调用 Agent
"""
import uuid

import streamlit as st

from agent.react_agent import ReactAgent
from messageMemory.file_history_store import get_history

# ==================== 1. 页面基础配置与样式 ====================
st.set_page_config(page_title="智扫通 · 智能客服", page_icon="🤖")

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}

    h1 {
        background: linear-gradient(90deg, #2b5876, #4e4376);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 10px 14px;
        margin-bottom: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    [data-testid="stExpander"] {
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        background: #fafbfc;
    }

    [data-testid="stSidebar"] {background: #f7f9fc;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 智扫通机器人智能客服")
st.caption("基于 LangChain ReAct Agent + RAG 检索增强")
st.divider()

# ==================== 2. 会话状态初始化 ====================
# Agent 实例：整个浏览器会话复用同一个，避免每次 rerun 重建
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

# 会话 ID：首次加载时生成，用于隔离不同浏览器会话的聊天历史
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"session_{uuid.uuid4().hex}"

# 界面消息缓存：仅用于页面渲染，真正的历史持久化由 ReactAgent 负责
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ==================== 3. 侧边栏：会话管理 ====================
with st.sidebar:
    st.subheader("会话管理")
    st.caption(f"会话 ID：{st.session_state['session_id']}")

    # 清空当前会话的历史文件 + 界面缓存
    if st.button("清空对话历史"):
        get_history(st.session_state["session_id"]).clear()
        st.session_state["messages"] = []
        st.rerun()


# ==================== 4. 渲染工具函数 ====================
# 工具函数名 → 用户可读的中文描述，避免内部实现细节暴露到界面
TOOL_LABELS = {
    "rag_summarize": "知识库检索",
    "get_weather": "天气查询",
    "get_user_location": "位置获取",
    "get_user_id": "用户信息",
    "get_current_month": "当前月份",
    "fetch_external_data": "使用记录查询",
    "fill_context_for_report": "报告准备",
}


def humanize(text: str) -> str:
    """把文本里的工具函数名替换成中文描述"""
    for name, label in TOOL_LABELS.items():
        text = text.replace(name, label)
    return text


def render_thinking(items: list[dict]) -> None:
    """把思考过程和查询结果折叠展示"""
    if not items:
        return
    labels = {"thinking": "思考", "tool": "查询结果"}
    with st.expander("🧠 思考过程"):
        for item in items:
            st.markdown(f"**{labels.get(item['type'], item['type'])}：** {humanize(item['content'])}")


def char_generator(text: str):
    """逐字输出，用于 st.write_stream"""
    for char in text:
        yield char


# ==================== 5. 渲染历史消息 ====================
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        render_thinking(message.get("thinking", []))
        st.markdown(message["content"])

# ==================== 6. 接收用户输入 ====================
prompt = st.chat_input()

# ==================== 7. 调用 Agent 并流式输出 ====================
if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    thinking_items: list[dict] = []
    final_text = ""

    with st.chat_message("assistant"):
        # 思考过程实时写入折叠框，结束后自动收起
        status = st.status("🧠 思考中...", expanded=False)
        with status:
            res_stream = st.session_state["agent"].execute_stream(
                prompt,
                st.session_state["session_id"],
            )
            for item in res_stream:
                if item["type"] == "final":
                    final_text += item["content"]
                else:
                    thinking_items.append(item)
                    label = {"thinking": "思考", "tool": "查询结果"}.get(item["type"], item["type"])
                    st.markdown(f"**{label}：** {humanize(item['content'])}")
        status.update(label="🧠 思考完成", state="complete", expanded=False)

        # 最终回答逐字输出
        if final_text:
            st.write_stream(char_generator(final_text))

    # 写入界面缓存，rerun 后由历史渲染逻辑复现
    st.session_state["messages"].append({
        "role": "assistant",
        "content": final_text,
        "thinking": thinking_items,
    })
    st.rerun()
