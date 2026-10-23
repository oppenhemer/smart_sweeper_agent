# 在线会话历史存储
# 手动实现FilecharMessageHistory
# 导入消息转字典和 字典转消息
import os
import json
from typing import Sequence
from utils.config_handler import rag_conf
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import messages_to_dict, messages_from_dict, BaseMessage

from utils.path_tool import get_abs_path


def get_history(session_id):
    return FileCharacterMessageHistory(session_id, get_abs_path(rag_conf["message_history_path"]))  # 返回一个聊天历史对象


class FileCharacterMessageHistory(BaseChatMessageHistory):
    # 一个会话id就是单独的一个文件
    def __init__(self, session_id, storage_path: str):
        self.session_id = session_id  # 会话id
        self.storage_path = storage_path  # 存储路径
        self.file_path = os.path.join(self.storage_path, self.session_id)
        os.makedirs(self.storage_path, exist_ok=True)  # 确保存储路径存在

    # message方法
    @property
    def messages(self) -> list[BaseMessage]:
        # 获取消息
        # 1.从存储路径中读取消息
        message_history = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                # json_load 将json文件转换为字典 messages_from_dict转化未消息
                message_history = messages_from_dict(json.load(f))
            return message_history
        except FileNotFoundError:
            return message_history

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # 添加到messages中
        all_messages = list(self.messages)  # 已有的消息列表
        all_messages.extend(messages)  # 新老融合消息
        # 1.将消息转字典
        message_dict = messages_to_dict(all_messages)
        # 2.将字典根据会话id写入文件
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(message_dict, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        # 清空文件
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
