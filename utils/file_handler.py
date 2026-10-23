# 导入md5模块
import hashlib
# 导入os模块,用于解析文件后缀
import os
# 导入langchain中pdf加载器
from typing import List
# 导入日志模块
from utils.logger_handler import logger

from langchain_community.document_loaders import PyPDFLoader
# 导入langchain中text加载器
from langchain_community.document_loaders import TextLoader
# 导入文档分隔 RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 按文本模式计算md5的文件后缀白名单（其余一律按二进制原始字节计算）
from utils.config_handler import load_chroma_config
from utils.path_tool import get_abs_path

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json"}
# 导入配置文件
chroma_conf = load_chroma_config()


# 获取文件的md5的十六进制字符串（自动按后缀选择文本/二进制模式）
def get_file_md5_hex(file_path, encoding='utf-8', chunk_size=8192):
    # 判断文件是否存在
    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件{file_path}不存在")
        return None
    # 判断是否是是文件
    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]路径{file_path}不是文件")
        return None
    # 初始化md5对象
    md5 = hashlib.md5()

    # 避免文件太大，设置分块大小
    chunk_size = 4096
    # 流式更新md5
    try:
        with open(file_path, 'rb') as file:
            chunk = file.read(chunk_size)
            while chunk:
                md5.update(chunk)
                chunk = file.read(chunk_size)
        return md5.hexdigest()
    except Exception as e:
        logger.error(f"[md5计算]文件{file_path}失败:{str(e)}")
        return None


# 返回文件夹内的文件列表（允许的文件后缀）
def listdir_with_allowed_type(dir_path: str, allow_type: tuple[str]) -> tuple[str]:
    files = []
    # 判断是否是文件夹
    if not os.path.isdir(dir_path):
        logger.error(f"[文件夹内文件列表]路径{dir_path}不是文件夹")
        return files
    # 获取文件夹中文件 endswith判断一个字符串是否以指定内容结尾,返回布尔值:
    for file in os.listdir(dir_path):
        if file.endswith(allow_type):
            # 补全路径
            files.append(os.path.join(dir_path, file))

    return tuple(files)


def pdf_loader(file_path, password=None) -> List[Document]:
    return PyPDFLoader(file_path, password=password).load()


def txt_loader(file_path) -> List[Document]:
    return TextLoader(file_path, encoding='utf-8').load()
