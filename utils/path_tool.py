"""
为整个工程提供统一的绝对路径
"""

import os


# 获取项目根目录
def get_project_root() -> str:
    # 获取当前文件绝对路径
    current_path = os.path.abspath(__file__)
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(current_path))
    return project_root


# 获取文件绝对路径
def get_abs_path(relative_path: str) -> str:
    # 获取项目根目录
    project_root = get_project_root()
    # 获取文件绝对路径
    file_absolute_path = os.path.join(project_root, relative_path)
    return file_absolute_path


# # 测试
# if __name__ == "__main__":
#     print(get_project_root())
#     print(get_file_absolute_path("utils\path_tool.py"))
