"""Reflexion Agent 手动搭建示例。

核心思想：执行 → 反思 → 根据反馈重新执行，直到结果满意或达到最大轮数。
适合需要自我改进、一次未必答对的任务（如跨城市对比、多条件分析）。

流程：actor(执行+调工具) → reflect(评估+反馈) ⇄ 条件边判断是否重试 → END。
"""

import json
import urllib.parse
import urllib.request
from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel, Field

from llm.deepseek.global_setting import make_llm

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


TOOLS = [get_weather]
_TOOL_BY_NAME = {t.name: t for t in TOOLS}

MAX_ITERATIONS = 3


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration: int = 0
    satisfied: bool = False


# ---------------------------------------------------------------------------
# actor：执行任务，调用工具给出尝试性回答
# ---------------------------------------------------------------------------
_actor_llm = make_llm().bind_tools(TOOLS)


def actor(state: State) -> dict:
    response = _actor_llm.invoke(
        [SystemMessage(content="你是执行者。请调用工具完成用户的要求并给出结果。")]
        + state.messages
    )

    out = [response]
    for tc in response.tool_calls:
        result = _TOOL_BY_NAME[tc["name"]].invoke(tc["args"])
        out.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # 拿到工具结果后，再调用一次 LLM 生成文字回答
    final = _actor_llm.invoke(
        [SystemMessage(content="基于工具结果，用简洁的语言回答用户最初的问题。")] + out
    )
    out.append(final)

    return {"messages": out, "iteration": state.iteration + 1}


# ---------------------------------------------------------------------------
# reflect：评估上一步结果是否满意，不满则给出具体反馈并触发重试
# ---------------------------------------------------------------------------
class Reflection(BaseModel):
    satisfied: bool = Field(description="结果是否已经完整、正确地满足用户最初的请求")
    feedback: str = Field(description="不满意时的具体改进建议；满意则为空字符串")


_reflector_llm = make_llm().with_structured_output(Reflection, method="function_calling")


def reflect(state: State) -> dict:
    reflection = _reflector_llm.invoke(
        [
            SystemMessage(
                content="你是反思者。请评估最近一次执行结果是否完整、正确地满足了用户最初的请求，"
                "并输出是否满意、以及不满意时的具体改进建议。"
            )
        ]
        + state.messages
    )

    messages = []
    if not reflection.satisfied:
        messages.append(
            HumanMessage(content=f"反思反馈：{reflection.feedback}\n请根据反馈重新执行并完善。")
        )
    return {"messages": messages, "satisfied": reflection.satisfied}


# ---------------------------------------------------------------------------
# 条件边：满意或超过最大轮数则结束，否则回到 actor 重试
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(State)
    builder.add_node("actor", actor)
    builder.add_node("reflect", reflect)

    builder.add_edge(START, "actor")
    builder.add_edge("actor", "reflect")

    def route(state: State) -> str:
        if state.satisfied or state.iteration >= MAX_ITERATIONS:
            return "end"
        return "actor"

    builder.add_conditional_edges("reflect", route, {"actor": "actor", "end": END})
    return builder.compile()


def main():
    graph = build_graph()
    question = "对比北京和上海的天气，告诉我哪个城市更适合今天外出散步，并说明理由"
    print(f"提问：{question}\n")

    result = graph.invoke({"messages": [HumanMessage(content=question)]})

    print("执行轨迹：")
    for m in result["messages"]:
        if isinstance(m, ToolMessage):
            print(f"  [工具] {m.content}")
        elif isinstance(m, HumanMessage) and m.content.startswith("反思反馈"):
            print(f"  [反思] {m.content.splitlines()[0]}")
    print(f"\n共迭代 {result['iteration']} 轮，最终满意：{result['satisfied']}\n")

    print("最终回答：")
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            print(m.content)
            break


if __name__ == "__main__":
    main()