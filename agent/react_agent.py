"""
React Agent
初始化Agent
执行
"""
from langchain.agents import create_agent
# 导入model
from agent.middleware import log_before_model, monitor_tool, report_prompt_switch
from messageMemory.file_history_store import FileCharacterMessageHistory, get_history
from model.factory import chat_model
# 加载提示词
from tools.agent_tools import get_user_location, rag_summarize, get_user_id, fetch_external_data, \
    fill_context_for_report, get_current_month, get_weather
from utils.prompt_loader import load_system_prompt
# 导入消息对象
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            tools=[rag_summarize, get_user_location, get_user_id, get_current_month, fetch_external_data,
                   fill_context_for_report, get_weather],
            system_prompt=load_system_prompt(),
            middleware=[log_before_model, monitor_tool, report_prompt_switch]
        )

    def execute_stream(self, query: str, session_id: str = "default"):
        # 拿到聊天历史对象
        store: FileCharacterMessageHistory = get_history(session_id)
        history = []  # List[BaseMessage]  # 聊天历史
        for m in store.messages:
            # 其他类型（ToolMessage、SystemMessage 等）直接跳过
            if isinstance(m, HumanMessage):
                history.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage) and not m.tool_calls:
                history.append({"role": "assistant", "content": m.content})
        # 拼接输入,agent每次都是从0开始的,所以必须查看历史消息
        input_dict = {
            "messages": history + [
                {"role": "user", "content": query},
            ]
        }
        lasted_content = ""
        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]

            if isinstance(latest_message, AIMessage):
                content = latest_message.content.strip() if isinstance(latest_message.content, str) else str(latest_message.content)
                if not content:
                    continue
                if latest_message.tool_calls:
                    # 带 tool_calls 的是"准备调工具"的过渡话术，归入思考过程
                    yield {"type": "thinking", "content": content}
                else:
                    # 不带 tool_calls 的是最终回答
                    lasted_content = content
                    yield {"type": "final", "content": content}
            elif isinstance(latest_message, ToolMessage):
                content = latest_message.content if isinstance(latest_message.content, str) else str(latest_message.content)
                yield {"type": "tool", "content": content}
        # 用户输入和模型最终回答持久化
        if lasted_content:
            store.add_messages((HumanMessage(content=query),
                                AIMessage(content=lasted_content)))


if __name__ == "__main__":
    agent = ReactAgent()
    for q in ["我在杭州，扫地机器人怎么保养", "那耗材多久换一次"]:
        print(f"\n===== 用户：{q} =====")
        for item in agent.execute_stream(q, "session_test"):
            print(f"[{item['type']}] {item['content']}", flush=True)
