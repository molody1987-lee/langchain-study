# -*- coding: utf-8 -*-
"""LangGraph 流程图渲染公共工具。

把编译后的图渲染成 graphviz 流程图（打印 DOT 源码 + 保存 .gv 文件 + 渲染图片），
供各示例脚本复用，避免每个脚本重复实现 render_graphviz。

依赖：
    pip install graphviz
    # 渲染成 PNG/SVG 图片还需要系统安装 Graphviz（提供 dot 命令）
    # macOS: brew install graphviz
    # 提示：未安装 graphviz 包时会自动回退为 LangGraph 内置的 mermaid 源码。
"""

import os

from langgraph.graph import END, START

# 输出目录：默认为本模块所在目录（graph/），可用环境变量 GRAPH_OUTPUT_DIR 覆盖
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def render_graphviz(graph, output_name, output_format="png"):
    """把 LangGraph 编译图转换成 graphviz 流程图并打印/渲染。

    - 始终打印 DOT 源码（不依赖系统 dot 命令）
    - 未安装 graphviz 包时，回退为 LangGraph 内置 mermaid 源码
    - 尝试渲染成图片文件（需要系统安装 Graphviz 提供 dot 命令）
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

    # 节点：START/END 使用特殊形状与友好标签
    shape_map = {START: "circle", END: "doublecircle"}
    label_map = {START: "START", END: "END"}
    for node_id in drawable.nodes:
        dot.node(node_id, label=label_map.get(node_id, node_id), shape=shape_map.get(node_id, "box"))

    # 边：条件边用虚线表示
    for edge in drawable.edges:
        label = edge.data if edge.data else ""
        if edge.conditional:
            dot.edge(edge.source, edge.target, label=label, style="dashed")
        else:
            dot.edge(edge.source, edge.target, label=label)

    print("=" * 60)
    print(f"DOT 源码（graphviz）: {output_name}")
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