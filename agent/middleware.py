"""
中间件文件：为 Agent 提供三类横切能力
1. monitor_tool       —— 工具调用监控：记录工具名、参数、耗时、异常
2. log_before_model   —— 模型调用监控：记录调用阶段、上下文规模、工具执行次数
3. report_prompt_switch —— 报告场景提示词切换：报告意图触发后改用报告提示词
"""
# 导入日志工具
import time
from typing import Callable
# 导入加载提示词工具
from utils.prompt_loader import load_report_prompt, load_system_prompt
from langchain_core.messages import ToolMessage, AIMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

from utils.logger_handler import logger

# 工具监视
from langchain.agents.middleware import before_model, wrap_tool_call, dynamic_prompt


@wrap_tool_call
def monitor_tool(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]):
    # 记录工具调用日志
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})

    # 取出模型决定调用工具时的思考内容（state 中最后一条带 tool_calls 的 AIMessage）reversed倒序遍历
    thinking = ""
    for msg in reversed((request.state or {}).get("messages", [])):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # content 正常是字符串；但多模态模型可能返回 list（文字 + 图片），直接 str() 兜底，避免后面日志格式化报错。
            thinking = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if thinking:
        logger.info(f"[模型思考]：{thinking}")
    logger.info(f"调用工具{tool_name}，参数：{tool_args}")
    start = time.time()
    try:
        if tool_name == "fill_context_for_report":
            # 若需要调用报告提示词，则设置标志位
            request.runtime.context['report_prompt_switch'] = True
        result = handler(request)
    except Exception as e:
        logger.error(f"工具{tool_name}调用失败，耗时{time.time() - start:.2f}秒，"
                     f"异常：{type(e).__name__}: {str(e)}")
        raise
    logger.info(f"工具{tool_name}调用成功，耗时{time.time() - start:.2f}秒")
    return result


# 模型调用前日志记录
@before_model
def log_before_model(state, runtime):
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    # 判断当前处于哪一阶段
    if last is None:
        stage = "初始调用"
    elif isinstance(last, ToolMessage):
        stage = "消化工具结果"
    else:
        stage = "处理用户输入"
    tool_count = sum(1 for m in messages if isinstance(m, ToolMessage))
    logger.info(f"[模型监控]开始调用模型，阶段：{stage}，"
                f"上下文消息数：{len(messages)}，已执行工具次数：{tool_count}")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None


# 当用户生成报告的意图时， 动态切换报告提示词
@dynamic_prompt
def report_prompt_switch(request):
    is_report_prompt_switch = request.runtime.context.get("report_prompt_switch", False)
    if is_report_prompt_switch:
        return load_report_prompt()
    return load_system_prompt()
