"""LangGraph stream_mode="values" 示例。

演示 values 模式：每一步（super-step）结束后产出「完整状态快照」。
适合调试时观察状态随图推进的完整变化过程（能看到所有字段的当前值）。

流程：
    START → refine_topic → generate_joke → END

运行（项目根目录）：
    python -m graph.graph_stream_values_test
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
# 4. values 模式：每步输出完整状态
# ---------------------------------------------------------------------------
def main():
    graph = build_graph()

    print("stream_mode=\"values\" —— 每一步后的完整状态快照：")
    for step, chunk in enumerate(graph.stream({"topic": "冰淇淋"}, stream_mode="values")):
        # chunk 是完整 State：能看到所有字段当前值（未设置的字段为默认值）
        print(f"  [Step {step}] {chunk}")

    # 打印流程图
    render_graphviz(graph, output_name="graph_stream_values_test")


if __name__ == "__main__":
    main()