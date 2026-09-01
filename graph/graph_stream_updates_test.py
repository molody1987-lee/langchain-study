"""LangGraph stream_mode="updates" 示例。

演示 updates 模式：每一步只产出「增量更新」（被该节点修改的字段），
并带上节点名。相比 values 更省带宽，适合实时显示「哪个节点改了哪些字段」。

流程：
    START → refine_topic → generate_joke → END

运行（项目根目录）：
    python -m graph.graph_stream_updates_test
"""

from pydantic import BaseModel

from langgraph.graph import END, START, StateGraph

from graph.graph_utils import render_graphviz


# ---------------------------------------------------------------------------
# 1. 状态定义
# ---------------------------------------------------------------------------
class State(BaseModel):
    topic: str = ""   # 主题（会被 refine_topic 修改）
    joke: str = ""    # 生成的段子（由 generate_joke 填充）


# ---------------------------------------------------------------------------
# 2. 节点函数
# ---------------------------------------------------------------------------
def refine_topic(state: State) -> dict:
    """第一步：扩写主题。"""
    return {"topic": state.topic + " 和猫"}


def generate_joke(state: State) -> dict:
    """第二步：基于主题生成一句话。"""
    return {"joke": f"关于「{state.topic}」的一个冷笑话"}


# ---------------------------------------------------------------------------
# 3. 建图
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(State)
    builder.add_node("refine_topic", refine_topic)
    builder.add_node("generate_joke", generate_joke)

    builder.add_edge(START, "refine_topic")
    builder.add_edge("refine_topic", "generate_joke")
    builder.add_edge("generate_joke", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# 4. updates 模式：每步输出增量更新
# ---------------------------------------------------------------------------
def main():
    graph = build_graph()

    print("stream_mode=\"updates\" —— 每一步的增量更新：")
    for step, chunk in enumerate(graph.stream({"topic": "冰淇淋"}, stream_mode="updates")):
        # chunk 形如 {"节点名": {"字段": "新值"}}，只包含被改动的字段
        for node_name, update in chunk.items():
            print(f"  [Step {step}] 节点 {node_name!r} 更新了: {update}")

    # 打印流程图
    render_graphviz(graph, output_name="graph_stream_updates_test")


if __name__ == "__main__":
    main()