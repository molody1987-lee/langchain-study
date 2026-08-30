"""ReWOO（Reasoning WithOut Observation）手动搭建示例。

与 Plan-and-Execute 的关键区别：
- Plan-and-Execute：逐步规划 → 每执行一步都「观察」结果再决定下一步。
- ReWOO：一次性把整个计划（含工具与参数）规划好 → 执行阶段不做任何观察/重规划，
  直接把所有工具按计划跑完 → 最后统一汇总。

优点：减少 LLM 调用次数（执行阶段不调用 LLM），速度快、成本低；
缺点：无法根据中间结果动态调整，适合步骤清晰、依赖可预先确定的确定性任务。

流程：planner(产出完整计划) → worker(按计划纯执行工具，无观察) → solver(汇总回答)。
"""

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Annotated, Any

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
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


_TOOL_BY_NAME = {t.name: t for t in [get_weather, get_current_time]}


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[dict[str, Any]] = []   # 完整计划：每个元素 {tool, args}
    results: list[str] = []           # 每个工具的执行结果


# ---------------------------------------------------------------------------
# planner：一次性产出完整计划（工具 + 参数 + 依赖占位符 #E{i}）
# ---------------------------------------------------------------------------
class PlanStep(BaseModel):
    tool: str = Field(description="要调用的工具名")
    args: dict[str, str] = Field(description="工具参数；参数值可引用前面结果，用 #E{i} 表示第 i 个结果")


class Plan(BaseModel):
    steps: list[PlanStep]


_planner_llm = make_llm().with_structured_output(Plan, method="function_calling")

_PLANNER_PROMPT = SystemMessage(
    content=(
        "你是一个规划器。请一次性把用户任务拆解成需要调用工具的完整步骤清单，"
        "每个步骤指定工具名和参数。若某步骤参数依赖前面步骤的结果，用 #E{i} 占位符引用"
        "第 i 个步骤的结果（i 从 0 开始）。不要执行，只输出计划。"
    )
)


def planner(state: State) -> dict:
    plan = _planner_llm.invoke([_PLANNER_PROMPT] + state.messages)
    return {"plan": [s.model_dump() for s in plan.steps]}


# ---------------------------------------------------------------------------
# worker：按计划纯执行，不调用 LLM、不做观察，只把 #E{i} 占位符替换成实际结果
# ---------------------------------------------------------------------------
def worker(state: State) -> dict:
    results: list[str] = []
    for step in state.plan:
        args = {}
        for k, v in step["args"].items():
            # 用之前的结果替换占位符 #E{i}
            args[k] = re.sub(
                r"#E(\d+)",
                lambda m: results[int(m.group(1))] if int(m.group(1)) < len(results) else m.group(0),
                str(v),
            )
        result = _TOOL_BY_NAME[step["tool"]].invoke(args)
        results.append(result)
    return {"results": results}


# ---------------------------------------------------------------------------
# solver：统一汇总所有结果，回答最初问题（唯一在执行阶段的 LLM 调用）
# ---------------------------------------------------------------------------
def solver(state: State) -> dict:
    context = "\n".join(f"[步骤 {i}] {r}" for i, r in enumerate(state.results))
    answer = make_llm().invoke(
        [
            SystemMessage(content="基于以下执行结果，回答用户最初的问题。"),
            HumanMessage(content=f"执行结果：\n{context}"),
        ]
        + state.messages
    )
    return {"messages": [answer]}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_node("worker", worker)
    builder.add_node("solver", solver)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "worker")
    builder.add_edge("worker", "solver")
    builder.add_edge("solver", END)

    return builder.compile()


def main():
    graph = build_graph()
    question = "北京现在天气怎么样？再告诉我现在几点"
    print(f"提问：{question}\n")

    result = graph.invoke({"messages": [HumanMessage(content=question)]})

    print("规划出的计划：")
    for i, s in enumerate(result["plan"]):
        print(f"  {i}. 工具 {s['tool']}, 参数 {s['args']}")

    print("\n执行结果：")
    for i, r in enumerate(result["results"]):
        print(f"  [步骤 {i}] {r}")

    print("\n最终回答：")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()