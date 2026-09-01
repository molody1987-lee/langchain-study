"""LangGraph 循环（loop）模式示例。

演示如何在图里形成「循环」：一个节点通过条件边指回自己，反复执行直到满足退出条件。
这也是 Agent 循环（LLM 反复调用工具直到不再调用）的底层机制。

流程：
    START → step ──(continue)──▶ step（指回自己，形成循环）
                    └(stop)────▶ END
"""

import operator
from typing import Annotated, Literal

from pydantic import BaseModel

from langgraph.graph import END, START, StateGraph

from graph.graph_utils import render_graphviz


# ---------------------------------------------------------------------------
# 1. 状态：counter 用普通字段（读后自增），log 用 reducer 跨轮累加
# ---------------------------------------------------------------------------
class State(BaseModel):
    counter: int = 0                               # 当前已循环轮数
    max_iters: int = 5                             # 循环上限（防止死循环）
    log: Annotated[list[str], operator.add] = []   # 每轮记录（累加）


# ---------------------------------------------------------------------------
# 2. 循环体节点
# ---------------------------------------------------------------------------
def node_step(state: State) -> dict:
    """单轮处理：计数 +1，并把本轮记录追加到 log。"""
    n = state.counter + 1
    return {"counter": n, "log": [f"第 {n} 轮处理"]}


# ---------------------------------------------------------------------------
# 3. 条件边：决定「继续循环」还是「退出」
# ---------------------------------------------------------------------------
def should_continue(state: State) -> Literal["continue", "stop"]:
    if state.counter < state.max_iters:
        return "continue"
    return "stop"


def build_graph():
    builder = StateGraph(State)
    builder.add_node("step", node_step)

    builder.add_edge(START, "step")
    # 关键：step 的一个分支指回自己（循环），另一个分支结束
    builder.add_conditional_edges(
        "step",
        should_continue,
        {"continue": "step", "stop": END},
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# 4. 运行示例
# ---------------------------------------------------------------------------
def main():
    graph = build_graph()

    result = graph.invoke({"max_iters": 5})
    print(f"循环共执行 {result['counter']} 轮：")
    for line in result["log"]:
        print("  ", line)

    # 渲染流程图
    render_graphviz(graph, output_name="graph_loop_test")


if __name__ == "__main__":
    main()