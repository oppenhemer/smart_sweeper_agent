"""
RAG总结服务
首先将用户提问检索资料，如何将用户提问和总结提交给模型，让模型总结给出最终结果
1.加载提示词文本
2.构造基础链
3.返回链
4.利用链检索文本
5.总结返回结果
"""
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from rag.vector_store import VectorStoreService

# 导入聊天提示词模板
from langchain_core.prompts import PromptTemplate
# 导入模型
from model.factory import chat_model
# 导入提示词加载
from utils.prompt_loader import load_rag_prompt
# 导入字符串解析器
from langchain_core.output_parsers import StrOutputParser
from messageMemory.file_history_store import FileCharacterMessageHistory, get_history
# 导入配置文件
from utils.config_handler import rag_conf


# 读取会话历史：只保留用户提问与最终回答，拼成纯文本
def _get_history_text(session_id: str) -> str:
    # 无会话标识时直接返回占位文本
    if not session_id:
        return "无"
    # 获取聊天历史对象
    store: FileCharacterMessageHistory = get_history(session_id)
    # 按消息条数截断，约等于最近 6 轮问答,截取最近消息,有几条给几条，最多12条
    recent_messages: List[BaseMessage] = store.messages[-rag_conf["rag_history_window"]:]  # 返回List[BaseMessage]
    lines = []
    for message in recent_messages:
        if isinstance(message, HumanMessage):
            lines.append(f"用户：{message.content}")
        # 带 tool_calls 的是过渡话术，不是用户看到过的回答，必须过滤
        elif isinstance(message, AIMessage) :
            lines.append(f"助手：{message.content}")
            # \n当分隔符，将列表拼接成字符串
    return "\n".join(lines) if lines else "无"


def temp(prompt):
    print("*" * 20)
    print(prompt.to_string())
    print("*" * 20)
    return prompt


class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.model = chat_model
        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.chain = self._init_chain()

    def rag_summarize(self, query: str, session_id: str):
        # 获取检索资料
        context_docs = self.retrieve_docs(query)
        # 组装成文本
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
            # 读取最近窗口的历史对话，仅用于理解指代与语境
        history_text = _get_history_text(session_id)
        return self.chain.invoke({"input": query, "context": context, "history": history_text})

    # 返回使用链
    def _init_chain(self):
        # temp 是调试透传函数，传函数对象本身，不要写成 temp()
        chain = self.prompt_template | temp | self.model | StrOutputParser()
        return chain

    # 检索文本
    def retrieve_docs(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)


# 测试
if __name__ == "__main__":
    rag_service = RagSummarizeService()
    print(rag_service.rag_summarize("那耗材多久换一次", "session_c2354ef4e731432583ee0300c0d708e7"))
