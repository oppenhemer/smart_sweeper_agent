# 提供模型
# 导入聊天模型
from abc import abstractmethod, ABC
from typing import Optional

from langchain_community.chat_models import ChatTongyi
# 导入嵌入模型
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
# 导入配置文件
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        return DashScopeEmbeddings(model=rag_conf["embed_model_name"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeModelFactory().generator()
