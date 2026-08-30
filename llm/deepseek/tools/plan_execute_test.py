"""Plan-and-Execute Agent 手动搭建示例。

对比 create_agent（ReAct 思维-行动交替），Plan-and-Execute 先让规划器把任务
拆成步骤清单，再逐步执行，最后汇总回答。适合结构清晰、可预先拆解的多步任务。

流程：planner(拆步骤) → executor(逐步调工具) ⇄ 条件边判断是否还有步骤 → final(汇总)。
"""

import json
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Annotated

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel, Field

from llm.deepseek.global_setting import make_llm

# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------
_WEATHER_URL = "https://wttr.in/{city}?format=j1"


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气情况。

    Args:
        city: 城市名称，如「北京」「上海」。
    """
    url = _WEATHER_URL.format(city=urllib.parse.quote(city))
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    current = data["current_condition"][0]
    desc = current.get("lang_zh", [{}])[0].get("value", current["weatherDesc"][0]["value"])
    return f"{city}：{desc}，温度 {current['temp_C']}℃，湿度 {current['humidity']}%"


@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [get_weather, get_current_time]
_TOOL_BY_NAME = {t.name: t for t in TOOLS}

# ---------------------------------------------------------------------------
# 图状态：在 messages 基础上增加计划列表和当前步骤指针
# ---------------------------------------------------------------------------
class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[str] = Field(default_factory=list)   # 待执行步骤
    current_step: int = 0                            # 当前执行到第几步


# ---------------------------------------------------------------------------
# 1) planner：把用户目标拆成步骤清单（结构化输出）
# ---------------------------------------------------------------------------
class Plan(BaseModel):
    steps: list[str] = Field(description="按顺序列出的执行步骤")


_planner_llm = make_llm().with_structured_output(Plan, method="function_calling")

_PLANNER_PROMPT = SystemMessage(
    content=(
        "你是一个任务规划器。请把用户的目标拆解成若干可逐步执行的具体步骤。\n"
        "要求：\n"
        "1. 每个步骤是单一、明确、需要调用工具才能完成的事实查询指令；\n"
        "2. 步骤之间按逻辑顺序排列；\n"
        "3. 不要包含「汇总」「报告」「回答用户」这类由后续节点负责的步骤。"
    )
)


def planner(state: State) -> dict:
    plan = _planner_llm.invoke([_PLANNER_PROMPT] + state.messages)
    return {"plan": plan.steps, "current_step": 0}


# ---------------------------------------------------------------------------
# 2) executor：执行当前这一个步骤（tool-calling LLM 决定调哪个工具）
# ---------------------------------------------------------------------------
_executor_llm = make_llm().bind_tools(TOOLS)


def executor(state: State) -> dict:
    step = state.plan[state.current_step]
    # 只针对当前这一步执行，不拼接历史，避免 LLM 看到全局后一次性做掉所有步骤
    response = _executor_llm.invoke(
        [SystemMessage(content=f"请只执行下面这个步骤，不要做其他步骤：\n{step}")]
    )

    out_messages = [response]
    # 手动执行 LLM 发出的工具调用，把结果作为 ToolMessage 加回
    for tc in response.tool_calls:
        tool_ = _TOOL_BY_NAME[tc["name"]]
        result = tool_.invoke(tc["args"])
        out_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return {"messages": out_messages, "current_step": state.current_step + 1}


# ---------------------------------------------------------------------------
# 3) final：汇总所有执行结果，回答最初的问题
# ---------------------------------------------------------------------------
_final_llm = make_llm()


def final_answer(state: State) -> dict:
    answer = _final_llm.invoke(
        [SystemMessage(content="根据上述步骤的执行结果，用简洁的语言回答用户最初的问题。")]
        + state.messages
    )
    return {"messages": [answer]}


# ---------------------------------------------------------------------------
# 条件边：还有步骤就继续 executor，否则进入 final
# ---------------------------------------------------------------------------
def should_continue(state: State) -> str:
    return "executor" if state.current_step < len(state.plan) else "final"


def build_graph():
    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("final", final_answer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_conditional_edges("executor", should_continue, {"executor": "executor", "final": "final"})
    builder.add_edge("final", END)

    return builder.compile()


def main():
    graph = build_graph()
    question = "帮我查一下北京和上海的天气，然后告诉我现在几点了"
    print(f"提问：{question}\n")

    result = graph.invoke({"messages": [HumanMessage(content=question)]})

    print("规划出的步骤：")
    for i, s in enumerate(result["plan"], 1):
        print(f"  {i}. {s}")

    print("\n执行轨迹：")
    for m in result["messages"]:
        if isinstance(m, HumanMessage):
            continue
        if isinstance(m, ToolMessage):
            print(f"  [工具结果] {m.content}")
        elif getattr(m, "tool_calls", None):
            print(f"  [调用工具] {[t['name'] for t in m.tool_calls]}")
        elif m.content:
            print(f"  [AI] {m.content[:100]}")

    print("\n最终回答：")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()