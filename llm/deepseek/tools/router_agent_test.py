"""Router（多 Agent 路由）手动搭建示例。

核心思想：一个路由节点先把用户请求分类，再转发给对应的「专业 Agent」处理。
适合把不同领域能力（天气、时间、数学、闲聊）拆分成独立专家，互不干扰。

流程：router(分类) → 条件边分发 → weather_agent / time_agent / math_agent / general_agent → END。
"""

import json
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel, Field

from llm.deepseek.global_setting import make_llm

_WEATHER_URL = "https://wttr.in/{city}?format=j1"


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气情况。"""
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


@tool
def calculate(expression: str) -> str:
    """计算一个数学表达式，如 '3*4+2'。仅支持 + - * / 和括号。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


_TOOL_BY_NAME = {t.name: t for t in [get_weather, get_current_time, calculate]}


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str = ""


# ---------------------------------------------------------------------------
# router：把用户请求分类到某个专业 Agent
# ---------------------------------------------------------------------------
class Route(BaseModel):
    target: Literal["weather", "time", "math", "general"] = Field(
        description="最合适的处理专家：weather 天气 / time 时间 / math 计算 / general 其他"
    )


_router_llm = make_llm().with_structured_output(Route, method="function_calling")


def router(state: State) -> dict:
    route = _router_llm.invoke(
        [SystemMessage(content="判断用户请求该交给哪个专业 Agent 处理。")] + state.messages
    )
    return {"route": route.target}


# ---------------------------------------------------------------------------
# 各专业 Agent：各自只绑定自己的工具，互不越权
# ---------------------------------------------------------------------------
def _run_expert(state: State, tools, instruction: str) -> dict:
    llm = make_llm().bind_tools(tools)
    response = llm.invoke(
        [SystemMessage(content=instruction)] + state.messages
    )
    out = [response]
    for tc in response.tool_calls:
        result = _TOOL_BY_NAME[tc["name"]].invoke(tc["args"])
        out.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    final = llm.invoke([SystemMessage(content="基于工具结果回答用户请求。")] + out)
    out.append(final)
    return {"messages": out}


def weather_agent(state: State) -> dict:
    return _run_expert(state, [get_weather], "你是天气专家，负责回答天气相关问题。")


def time_agent(state: State) -> dict:
    return _run_expert(state, [get_current_time], "你是时间专家，负责回答时间相关问题。")


def math_agent(state: State) -> dict:
    return _run_expert(state, [calculate], "你是数学专家，负责解决计算问题。")


def general_agent(state: State) -> dict:
    llm = make_llm()
    answer = llm.invoke(
        [SystemMessage(content="你是通用助手，直接回答用户的日常问题。")] + state.messages
    )
    return {"messages": [answer]}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("router", router)
    builder.add_node("weather_agent", weather_agent)
    builder.add_node("time_agent", time_agent)
    builder.add_node("math_agent", math_agent)
    builder.add_node("general_agent", general_agent)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        lambda s: s.route,
        {
            "weather": "weather_agent",
            "time": "time_agent",
            "math": "math_agent",
            "general": "general_agent",
        },
    )
    for node in ["weather_agent", "time_agent", "math_agent", "general_agent"]:
        builder.add_edge(node, END)

    return builder.compile()


def main():
    graph = build_graph()
    questions = [
        "北京今天天气怎么样？",
        "现在几点了？",
        "帮我算一下 23*17+4 等于多少？",
        "给我讲个冷笑话。",
    ]

    for q in questions:
        result = graph.invoke({"messages": [HumanMessage(content=q)]})
        route = result["route"]
        answer = result["messages"][-1].content
        print(f"提问：{q}")
        print(f"路由 → {route}")
        print(f"回答：{answer}\n{'=' * 50}\n")


if __name__ == "__main__":
    main()