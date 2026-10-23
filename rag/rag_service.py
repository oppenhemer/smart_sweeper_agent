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

from messageMemory.file_history_store import FileCharacterMessageHistory
from rag.vector_store import VectorStoreService

# 导入聊天提示词模板
from langchain_core.prompts import PromptTemplate
# 导入模型
from model.factory import chat_model
# 导入提示词加载
from utils.prompt_loader import load_rag_prompt
# 导入字符串解析器
from langchain_core.output_parsers import StrOutputParser


# 导入配置文件


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

    def rag_summarize(self, query: str):
        # 获取检索资料
        context_docs = self.retrieve_docs(query)
        # 组装成文本
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
        return self.chain.invoke({"input": query, "context": context})

    # 返回使用链
    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    # 检索文本
    def retrieve_docs(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)


# 测试
if __name__ == "__main__":
    rag_service = RagSummarizeService()
    print(rag_service.rag_summarize("小户型适合哪些扫地机器人"))
