# 导入chroma向量库
import os
from typing import List

from langchain_chroma import Chroma

# 导入文档分隔 RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 导入文件处理模块
from utils.file_handler import pdf_loader, txt_loader, get_file_md5_hex, listdir_with_allowed_type
# 导入配置文件
from utils.config_handler import chroma_conf
# 向量库服务类
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
# 导入模型
from model.factory import embed_model


class VectorStoreService:
    def __init__(self):
        self.vector_store = vectorstore = Chroma(
            collection_name=chroma_conf["collection_name"],  # 相当于设置数据库名
            embedding_function=embed_model,  # 使用的嵌入模型
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),  # 数据库位置
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separator"]
        )

    def load_document(self):
        """
        从数据文件夹中读取数据文件，转为向量存入向量
        同时还需要做MD5去重，已经在向量库中则不需要加载
        """

        def check_md5(md5):
            # 判断md5文件是否存在,不存在则创建
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 不存在则创建
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w").close()

            # 判断是否在md5.txt文件中
            # md5存在的话则读文档对比传入的md5
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5:
                        return True
                return False

        # 读取文件 获得文件元组  里面存的是文件路径
        file_list: tuple[str] = listdir_with_allowed_type(get_abs_path(chroma_conf["data_path"]),
                                                          tuple(chroma_conf["allow_knowledge_file_type"]))
        # 遍历文件元组
        for file_path in file_list:
            # 读文件,获取md5码 判断去重
            file_md5 = get_file_md5_hex(file_path)
            if file_md5 is None:
                logger.warning(f"[加载知识库]{file_path}md5计算失败，跳过")
                continue
            if not check_md5(file_md5):
                try:
                    # 不存在则加入向量库,同时需要判断文件类型，如何加载
                    if file_path.endswith(".pdf"):
                        docs = pdf_loader(file_path)
                    elif file_path.endswith(".txt"):
                        docs = txt_loader(file_path)
                        # 加载文档后切分
                    else:
                        continue
                    docs = self.spliter.split_documents(docs)
                    if not docs:
                        logger.warning(f"[加载知识库]{file_path}没有有效文本内容，跳过")
                        continue
                    # 加入向量库
                    self.vector_store.add_documents(docs)
                    logger.info(f"[加载知识库]{file_path}成功")
                    # 同时需要把md5加入md5.txt文件中
                    with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                        f.write(file_md5 + "\n")
                except Exception as e:
                    logger.warning(f"[加载知识库]{file_path}失败: {type(e).__name__}: {str(e)}")
            else:
                logger.info(f"[加载知识库]{file_path}已存在，跳过")

    # 获取检索器
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})


if __name__ == "__main__":
    v = VectorStoreService()
    v.load_document()
    retriever = v.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("*" * 20)
