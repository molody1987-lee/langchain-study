"""LangGraph Time Travel（时间旅行）示例。

演示利用 checkpointer 的 checkpoint 历史实现「时间旅行」：
    1. Replay（回放）：用 invoke(None, snapshot.config) 从某个历史 checkpoint 重新执行；
       该 checkpoint 之前的节点不会重跑，之后的节点会重新执行。
    2. Fork（分叉）：用 update_state 在历史 checkpoint 上改写状态后继续执行，
       从而在不破坏原线程历史的前提下探索另一条分支。

关键 API：
    - graph.compile(checkpointer=...)                    注入检查点（每个超级步骤保存一次状态）
    - graph.get_state_history(config)                    获取该 thread 全部 checkpoint 快照（新的在前）
    - graph.invoke(None, snapshot.config)                从某个 checkpoint 回放
    - graph.update_state(snapshot.config, values=...)    在某 checkpoint 上分叉并改写状态

流程：
    START → generate_topic → write_joke → END

运行（项目根目录）：
    python -m graph.graph_time_travel_test
"""

from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from graph.graph_utils import render_graphviz


# ---------------------------------------------------------------------------
# 1. 状态定义
# ---------------------------------------------------------------------------
class State(BaseModel):
    topic: str = ""   # 主题
    joke: str = ""    # 生成的段子


# ---------------------------------------------------------------------------
# 2. 节点函数（打印执行痕迹，便于观察 replay / fork 时哪一步被重跑）
# ---------------------------------------------------------------------------
def generate_topic(state: State) -> dict:
    print("  [执行] generate_topic")
    return {"topic": "烘干机里的袜子"}


def write_joke(state: State) -> dict:
    print("  [执行] write_joke")
    return {"joke": f"为什么{state.topic}会消失？因为它们私奔了！"}


# ---------------------------------------------------------------------------
# 3. 建图（注入 checkpointer）
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(State)
    builder.add_node("generate_topic", generate_topic)
    builder.add_node("write_joke", write_joke)

    builder.add_edge(START, "generate_topic")
    builder.add_edge("generate_topic", "write_joke")
    builder.add_edge("write_joke", END)

    return builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 4. 演示：完整运行 → 查看历史 → Replay → Fork
# ---------------------------------------------------------------------------
def _print_history(graph, config):
    """按时间倒序打印当前 thread 的 checkpoint 历史（新的在前）。"""
    print("checkpoint 历史（新的在前）：")
    for i, snap in enumerate(graph.get_state_history(config)):
        cid = snap.config["configurable"]["checkpoint_id"]
        print(f"  [{i}] checkpoint={cid}  next={snap.next}  values={snap.values}")
    print()


def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "1"}}

    print("== 第一次完整运行 ==")
    result = graph.invoke({}, config)
    print("最终结果：", result)
    _print_history(graph, config)

    # ---- Replay：回放 write_joke 这一步 ----
    history = list(graph.get_state_history(config))
    before_joke = next(s for s in history if s.next == ("write_joke",))
    print("== Replay：从 write_joke 之前的 checkpoint 回放（generate_topic 不重跑） ==")
    replay_result = graph.invoke(None, before_joke.config)
    print("回放结果：", replay_result)

    # ---- Fork：在 write_joke 之前改主题，分叉出新结果 ----
    print("== Fork：在 write_joke 之前把 topic 改为「鸡」再继续 ==")
    fork_config = graph.update_state(before_joke.config, values={"topic": "鸡"})
    fork_result = graph.invoke(None, fork_config)
    print("分叉结果：", fork_result)
    print("（分叉不改动原线程，原最终结果仍是：", result, "）")

    # 打印流程图
    render_graphviz(graph, output_name="graph_time_travel_test")


if __name__ == "__main__":
    main()