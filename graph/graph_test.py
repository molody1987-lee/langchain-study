# -*- coding: utf-8 -*-
"""
基于 LangGraph 的图流程示例，并使用 graphviz 包渲染流程图。

依赖安装：
    pip install langgraph graphviz
    # 渲染成 PNG/SVG 图片还需要系统安装 Graphviz（提供 dot 命令）
    # macOS: brew install graphviz
    # 提示：未安装 graphviz 包时会自动回退为 LangGraph 内置的 mermaid 源码。
"""

import os
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

# 脚本所在目录，用于定位渲染输出文件（可用环境变量 GRAPH_OUTPUT_DIR 覆盖）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1. 定义状态（State）：节点之间传递的数据结构
# ---------------------------------------------------------------------------
class State(TypedDict):
    text: str       # 原始输入文本
    category: str   # 分类结果
    result: str     # 最终处理结果


# ---------------------------------------------------------------------------
# 2. 定义节点函数：每个节点接收 state，返回要更新的部分字段
# ---------------------------------------------------------------------------
def node_generate(state: State) -> dict:
    """生成文本。"""
    return {"text": state.get("text", "hello") + " -> generated"}


def node_classify(state: State) -> dict:
    """对文本进行分类。"""
    text = state.get("text", "")
    return {"category": "positive" if "ok" in text else "negative"}


def node_handle_positive(state: State) -> dict:
    """正向分支处理。"""
    return {"result": f"[positive] processed: {state.get('text')}"}


def node_handle_negative(state: State) -> dict:
    """负向分支处理。"""
    return {"result": f"[negative] processed: {state.get('text')}"}


def route_by_category(state: State) -> Literal["handle_positive", "handle_negative"]:
    """条件路由：根据 category 决定下一个节点。"""
    return "handle_positive" if state.get("category") == "positive" else "handle_negative"


# ---------------------------------------------------------------------------
# 3. 构建图：添加节点、边、条件边
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(State)

    builder.add_node("generate", node_generate)
    builder.add_node("classify", node_classify)
    builder.add_node("handle_positive", node_handle_positive)
    builder.add_node("handle_negative", node_handle_negative)

    builder.add_edge(START, "generate")                 # 开始 -> generate
    builder.add_edge("generate", "classify")            # generate -> classify
    builder.add_conditional_edges(                      # classify -> (条件分支)
        "classify",
        route_by_category,
        {
            "handle_positive": "handle_positive",
            "handle_negative": "handle_negative",
        },
    )
    builder.add_edge("handle_positive", END)            # handle_positive -> 结束
    builder.add_edge("handle_negative", END)            # handle_negative -> 结束

    return builder.compile()


# ---------------------------------------------------------------------------
# 4. 使用 graphviz 包渲染流程图
# ---------------------------------------------------------------------------
def _node_label(node_id: str) -> str:
    """把内部节点 id 转成更友好的显示名称。"""
    return {START: "START", END: "END"}.get(node_id, node_id)


def render_graphviz(graph, output_name: str = "graph_test", output_format: str = "png"):
    """把 LangGraph 编译图转换成 graphviz 流程图并渲染。

    - 始终输出 DOT 源码（不依赖系统 dot 命令）
    - 尝试渲染成图片文件（需要系统安装 Graphviz）
    """
    drawable = graph.get_graph()

    try:
        from graphviz import Digraph
    except ImportError:
        print("[WARN] 未安装 graphviz 包，回退为 LangGraph 内置 mermaid 源码：")
        print(drawable.draw_mermaid())
        return None

    dot = Digraph(comment="LangGraph Flow", format=output_format)
    dot.attr(rankdir="LR")  # 从左到右布局

    # 添加节点（START/END 使用特殊形状）
    shape_map = {START: "circle", END: "doublecircle"}
    for node_id in drawable.nodes:
        dot.node(node_id, label=_node_label(node_id), shape=shape_map.get(node_id, "box"))

    # 添加边（条件边用虚线表示）
    for edge in drawable.edges:
        label = edge.data if edge.data else ""
        if edge.conditional:
            dot.edge(edge.source, edge.target, label=label, style="dashed")
        else:
            dot.edge(edge.source, edge.target, label=label)

    # 输出 DOT 源码（无需系统 dot 即可生成）
    print("=" * 60)
    print("DOT 源码（graphviz）:")
    print("=" * 60)
    print(dot.source)

    # 输出目录：默认为脚本所在目录，可用环境变量 GRAPH_OUTPUT_DIR 覆盖
    output_dir = os.environ.get("GRAPH_OUTPUT_DIR") or SCRIPT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_name)

    # 保存 DOT 文件（不依赖系统 Graphviz）
    dot.save(output_file + ".gv")
    print(f"DOT 文件已保存: {output_file}.gv")

    # 渲染成图片（需要系统安装 Graphviz 提供 dot 命令）
    try:
        rendered_path = dot.render(output_file, cleanup=True, view=False)
        print(f"流程图已保存: {rendered_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 渲染图片失败（可能未安装系统 Graphviz/dot）: {exc}")
        print("       可先将上方 DOT 源码或 .gv 文件用 dot 命令渲染。")

    return dot


def main():
    graph = build_graph()

    # 运行一次流程
    print("=" * 60)
    print("运行 LangGraph 流程:")
    print("=" * 60)
    result = graph.invoke({"text": "ok"})
    print(result)

    # 使用 graphviz 渲染流程图
    render_graphviz(graph, output_name="graph_test", output_format="png")


if __name__ == "__main__":
    main()