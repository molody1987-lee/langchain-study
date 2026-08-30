"""基于大模型的单 Agent 图示例。

用 LangGraph 手动搭建一个「单 Agent 循环」：一个 agent 节点绑定工具，
LLM 决定是否调用工具；若有工具调用则执行后回到 agent 继续，直到不再调用工具时结束。

流程：
    START → agent ⇄ 条件边(是否还有工具调用) → END
"""

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel

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


@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [get_weather, get_current_time]
_TOOL_BY_NAME = {t.name: t for t in TOOLS}

MAX_STEPS = 5  # 最多循环多少轮，防止死循环


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration: int = 0


# ---------------------------------------------------------------------------
# agent 节点：决定是否调用工具；调用则执行工具并把结果加回，等待下一轮
# ---------------------------------------------------------------------------
_agent_llm = make_llm().bind_tools(TOOLS)

_AGENT_PROMPT = SystemMessage(content="你是一个乐于助人的助手，可以查询天气和时间。")


def agent(state: State) -> dict:
    response = _agent_llm.invoke([_AGENT_PROMPT] + state.messages)

    out = [response]
    # 若 LLM 决定调用工具，执行后把结果作为 ToolMessage 加回
    for tc in response.tool_calls:
        result = _TOOL_BY_NAME[tc["name"]].invoke(tc["args"])
        out.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return {"messages": out, "iteration": state.iteration + 1}


# ---------------------------------------------------------------------------
# 条件边：还有工具调用就继续循环，否则结束
# ---------------------------------------------------------------------------
def should_continue(state: State) -> str:
    # 检查最近一条 AIMessage 是否发出了工具调用（工具结果 ToolMessage 不算）
    for m in reversed(state.messages):
        if isinstance(m, AIMessage):
            if m.tool_calls and state.iteration < MAX_STEPS:
                return "agent"
            return "end"
    return "end"


def build_graph():
    builder = StateGraph(State)
    builder.add_node("agent", agent)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"agent": "agent", "end": END})

    return builder.compile()


# ---------------------------------------------------------------------------
# 使用 graphviz 把编译后的图渲染成流程图
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def render_graphviz(graph, output_name: str = "agent_graph", output_format: str = "png"):
    """把 LangGraph 编译图转换成 graphviz 流程图并渲染。

    - 始终输出 DOT 源码（不依赖系统 dot 命令）
    - 尝试渲染成图片文件（需要系统安装 Graphviz 提供 dot 命令）
    """
    drawable = graph.get_graph()

    try:
        from graphviz import Digraph
    except ImportError:
        print("[WARN] 未安装 graphviz 包，回退为 LangGraph 内置 mermaid 源码：")
        print(drawable.draw_mermaid())
        return None

    dot = Digraph(comment="LangGraph Agent Flow", format=output_format)
    dot.attr(rankdir="LR")  # 从左到右布局

    shape_map = {START: "circle", END: "doublecircle"}
    for node_id in drawable.nodes:
        dot.node(node_id, label=node_id, shape=shape_map.get(node_id, "box"))

    for edge in drawable.edges:
        label = edge.data if edge.data else ""
        if edge.conditional:
            dot.edge(edge.source, edge.target, label=label, style="dashed")
        else:
            dot.edge(edge.source, edge.target, label=label)

    # 输出 DOT 源码
    print("=" * 60)
    print("DOT 源码（graphviz）:")
    print("=" * 60)
    print(dot.source)

    # 保存 .gv 文件（不依赖系统 Graphviz）
    output_dir = os.environ.get("GRAPH_OUTPUT_DIR") or SCRIPT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_name)
    dot.save(output_file + ".gv")
    print(f"DOT 文件已保存: {output_file}.gv")

    # 渲染成图片（需要系统安装 Graphviz / dot）
    try:
        rendered_path = dot.render(output_file, cleanup=True, view=False)
        print(f"流程图已保存: {rendered_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 渲染图片失败（可能未安装系统 Graphviz/dot）: {exc}")

    return dot


def main():
    graph = build_graph()
    question = "北京现在天气怎么样？再顺便告诉我现在几点"
    print(f"提问：{question}\n")

    result = graph.invoke({"messages": [HumanMessage(content=question)]})

    print("执行轨迹：")
    for m in result["messages"]:
        if isinstance(m, ToolMessage):
            print(f"  [工具结果] {m.content}")
        elif isinstance(m, AIMessage) and m.tool_calls:
            print(f"  [调用工具] {[t['name'] for t in m.tool_calls]}")

    print(f"\n共循环 {result['iteration']} 轮\n")
    print("最终回答：")
    print(result["messages"][-1].content)

    # 渲染流程图
    render_graphviz(graph, output_name="agent_graph", output_format="png")


if __name__ == "__main__":
    main()