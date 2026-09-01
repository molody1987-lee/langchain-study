"""
LangGraph 条件路由与选择示例。

覆盖两种「条件决策」核心能力：
    1. add_conditional_edges —— switch 式条件分支路由
       根据状态值，把一个节点的出口路由到多个不同分支之一，分支处理完后汇合。
    2. Send —— 动态并行扇出（条件选择多个）
       根据状态动态决定「并行发往哪些节点」，等所有发出去的节点都跑完后自动合流。

运行：
    python graph_switch_test.py
"""

import operator
import os
from typing import Annotated, Literal

from pydantic import BaseModel

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from graph.graph_utils import render_graphviz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# 演示一：add_conditional_edges —— switch 式条件分支路由
# ===========================================================================
class State(BaseModel):
    query: str = ""    # 用户输入
    intent: str = ""   # 分类结果（决定走哪条分支）
    result: str = ""   # 最终处理结果


def node_classify(state: State) -> dict:
    """意图分类：根据 query 内容打标签，标签将决定后续分支。"""
    q = state.query
    if "天气" in q or "weather" in q:
        return {"intent": "weather"}
    if "时间" in q or "几点" in q:
        return {"intent": "time"}
    if "翻译" in q or "translate" in q:
        return {"intent": "translate"}
    return {"intent": "fallback"}


def node_handle_weather(state: State) -> dict:
    return {"result": f"[天气] 为「{state.query}」查询天气"}


def node_handle_time(state: State) -> dict:
    return {"result": f"[时间] 为「{state.query}」返回当前时间"}


def node_handle_translate(state: State) -> dict:
    return {"result": f"[翻译] 为「{state.query}」执行翻译"}


def node_handle_fallback(state: State) -> dict:
    return {"result": f"[兜底] 无法识别「{state.query}」"}


def node_finalize(state: State) -> dict:
    """所有分支汇合后的公共收尾节点。"""
    return {"result": state.result + " ✔"}


def route_intent(state: State) -> Literal["handle_weather", "handle_time", "handle_translate", "handle_fallback"]:
    """条件路由函数：返回下一个节点名（类似 switch 的 case 分发）。

    返回值必须是图中的节点名（或 mapping 里的键），这里是「直接返回节点名」的写法。
    """
    if state.intent == "weather":
        return "handle_weather"
    if state.intent == "time":
        return "handle_time"
    if state.intent == "translate":
        return "handle_translate"
    return "handle_fallback"


def demo_conditional_routing():
    print("=" * 66)
    print("演示一：add_conditional_edges（switch 式条件分支路由）")
    print("=" * 66)

    builder = StateGraph(State)
    builder.add_node("classify", node_classify)
    builder.add_node("handle_weather", node_handle_weather)
    builder.add_node("handle_time", node_handle_time)
    builder.add_node("handle_translate", node_handle_translate)
    builder.add_node("handle_fallback", node_handle_fallback)
    builder.add_node("finalize", node_finalize)

    builder.add_edge(START, "classify")
    # 关键：从 classify 出发，按 route_intent 的返回值分发到不同分支
    builder.add_conditional_edges("classify", route_intent)
    # 四个分支处理完后汇合到同一个 finalize 节点
    builder.add_edge(["handle_weather", "handle_time", "handle_translate", "handle_fallback"], "finalize")
    builder.add_edge("finalize", END)

    # 注：也可以用「mapping 字典」风格，让路由函数返回一个「键」，再映射到节点：
    #   builder.add_conditional_edges(
    #       "classify",
    #       lambda state: state.intent,   # 返回键
    #       {"weather": "handle_weather", "time": "handle_time",
    #        "translate": "handle_translate", "fallback": "handle_fallback"},
    #   )

    graph = builder.compile()

    queries = ["今天天气怎么样", "现在几点了", "把 hello 翻译成中文", "来一段说唱"]
    for q in queries:
        result = graph.invoke({"query": q})
        print(f"query={q!r}")
        print(f"  intent={result['intent']!r:12} -> {result['result']}")

    # 用 stream 观察一次完整执行里「实际走了哪些节点」
    print("\nstream(updates) 观察「今天天气怎么样」的节点执行顺序:")
    for chunk in graph.stream({"query": "今天天气怎么样"}, stream_mode="updates"):
        print("  执行节点:", list(chunk.keys()))

    # 渲染流程图
    render_graphviz(graph, output_name="graph_switch_routing")


# ===========================================================================
# 演示二：Send —— 动态并行扇出（条件选择多个目标）
# ===========================================================================
class FanState(BaseModel):
    keywords: list[str] = []                        # 要分发的关键词
    commits: Annotated[list[str], operator.add] = []  # 并行结果用 reducer 累加


def node_dispatch(state: FanState) -> list[Send]:
    """分发器：根据 keywords 数量动态「并行」发往多个 worker 实例。

    返回 list[Send]，每个 Send 会各自开一个 worker 节点实例并行执行。
    注意：arg 直接传 dict 时，worker 收到的 state 是不带类型的 dict；
    传 FanState 模型实例，节点里才能用 state.keywords 属性访问。
    """
    return [Send("worker", FanState(keywords=[k])) for k in state.keywords]


def node_worker(state: FanState) -> dict:
    """每个并行实例拿到的 keywords 只有自己负责的那一个。"""
    k = state.keywords[0]
    return {"commits": [f"worker 处理了「{k}」"]}


def node_join(state: FanState) -> dict:
    """合流节点：等所有 Send 出去的 worker 都跑完后，才会执行这里。"""
    total = len(state.commits)
    return {"commits": [f"合计 {total} 个 worker 结果"]}


def demo_send_fanout():
    print()
    print("=" * 66)
    print("演示二：Send（动态并行扇出 / 条件选择多个）")
    print("=" * 66)

    builder = StateGraph(FanState)
    builder.add_node("worker", node_worker)
    builder.add_node("join", node_join)

    # 分发器挂在 START：返回 list[Send]，决定「并行派发几个 worker」
    builder.add_conditional_edges(START, node_dispatch)
    # worker 全部完成后自动进入 join（join 会等所有并行分支结束）
    builder.add_edge("worker", "join")
    builder.add_edge("join", END)

    graph = builder.compile()

    kws = ["天气", "时间", "翻译"]
    result = graph.invoke({"keywords": kws})
    print(f"keywords = {kws}")
    for line in result["commits"]:
        print(" ", line)

    # 渲染流程图（Send 动态扇出图需手动绘制，见 render_fanout_manual）
    render_fanout_manual(output_name="graph_switch_fanout")


def render_fanout_manual(output_name: str, output_format: str = "png"):
    """手动绘制 Send 扇出结构的流程图。

    说明：langgraph 1.2.x 的 get_graph() 对基于 Send 的动态扇出图无法正确枚举
    静态边（只会给出一条 dispatch -> end），因此这里手动绘制真实结构：
        START -> 动态派发(返回 list[Send]) -> worker ×N(并行) -> join -> END
    """
    from graphviz import Digraph

    dot = Digraph(comment="LangGraph Send Fanout", format=output_format)
    dot.attr(rankdir="LR")

    dot.node("__start__", label="__start__", shape="circle")
    dot.node("dispatch", label="动态派发\\n返回 list[Send]", shape="box")
    dot.node("worker", label="worker ×N\\n(并行)", shape="box")
    dot.node("join", label="join\\n(等所有 worker 完成)", shape="box")
    dot.node("__end__", label="__end__", shape="doublecircle")

    dot.edge("__start__", "dispatch")
    dot.edge("dispatch", "worker", label="Send", style="dashed")
    dot.edge("worker", "join")
    dot.edge("join", "__end__")

    print("=" * 60)
    print(f"DOT 源码（graphviz）: {output_name}")
    print("=" * 60)
    print(dot.source)

    output_dir = os.environ.get("GRAPH_OUTPUT_DIR") or SCRIPT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_name)
    dot.save(output_file + ".gv")
    print(f"DOT 文件已保存: {output_file}.gv")

    try:
        rendered_path = dot.render(output_file, cleanup=True, view=False)
        print(f"流程图已保存: {rendered_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 渲染图片失败（可能未安装系统 Graphviz/dot）: {exc}")

    return dot


def main():
    demo_conditional_routing()
    demo_send_fanout()


if __name__ == "__main__":
    main()