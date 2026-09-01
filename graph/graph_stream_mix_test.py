"""LangGraph 混合流模式示例。

演示 stream_mode 传「列表」同时启用多种流模式：["updates", "values", "messages"]。
此时每次产出的是一个 (mode, chunk) 元组，需要按 mode 判断 chunk 的类型：
    - values   -> 完整状态快照（dict）
    - updates  -> 某个节点的增量更新（{"节点名": {"字段": "新值"}}）
    - messages -> (message_chunk, metadata) 元组，逐 token 输出 LLM 文本

流程：
    START → refine_topic（非 LLM 节点，产出 values/updates）
          → call_model（LLM 节点，额外产出 messages 令牌）
          → END

运行（项目根目录）：
    python -m graph.graph_stream_mix_test
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from langgraph.graph import END, START, StateGraph

from graph.graph_utils import render_graphviz
from llm.deepseek.global_setting import make_llm


# ---------------------------------------------------------------------------
# 1. 状态定义
# ---------------------------------------------------------------------------
class State(BaseModel):
    topic: str = ""   # 主题（由 refine_topic 改写）
    joke: str = ""    # 生成的段子（由 call_model 填充）


# ---------------------------------------------------------------------------
# 2. 节点函数：一个普通节点 + 一个 LLM 节点
# ---------------------------------------------------------------------------
_llm = make_llm()
_SYSTEM_PROMPT = "你是一个幽默的段子手，回答要简短有趣。"


def refine_topic(state: State) -> dict:
    """普通节点：改写主题，只会在 values/updates 模式中出现。"""
    return {"topic": state.topic + " 和猫"}


def call_model(state: State) -> dict:
    """LLM 节点：逐 token 生成一句话，会在 messages 模式中出现。"""
    response = _llm.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"讲一个关于「{state.topic}」的冷笑话"),
        ]
    )
    return {"joke": response.content}


# ---------------------------------------------------------------------------
# 3. 建图
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(State)
    builder.add_node("refine_topic", refine_topic)
    builder.add_node("call_model", call_model)

    builder.add_edge(START, "refine_topic")
    builder.add_edge("refine_topic", "call_model")
    builder.add_edge("call_model", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# 4. 混合模式：同时流式输出 values / updates / messages
# ---------------------------------------------------------------------------
def main():
    graph = build_graph()

    print("混合模式 stream_mode=[\"updates\", \"values\", \"messages\"]：")
    for mode, chunk in graph.stream(
        {"topic": "程序员"},
        stream_mode=["updates", "values", "messages"],
    ):
        if mode == "messages":
            # messages 模式的 chunk 是 (message_chunk, metadata)，逐 token 打印
            token, _metadata = chunk
            if token.content:
                print(token.content, end="", flush=True)
        else:
            # values / updates 模式的 chunk 分别为完整状态、节点增量更新
            print(f"[{mode}] {chunk}")
    print()

    # 打印流程图
    render_graphviz(graph, output_name="graph_stream_mix_test")


if __name__ == "__main__":
    main()