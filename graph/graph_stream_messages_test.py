"""LangGraph stream_mode="messages" 示例。

演示 messages 模式：在调用 LLM 的节点中，逐 token 流式输出（打字机效果）。
每次产出的是一个元组 (message_chunk, metadata)：
    - message_chunk：LLM 生成的一段文本（通常是单个 token 或几段字符）
    - metadata     ：包含 langgraph_node、langgraph_step 等上下文信息

流程：
    START → call_model → END

运行（项目根目录）：
    python -m graph.graph_stream_messages_test
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
    topic: str = ""   # 段子主题
    joke: str = ""    # 最终生成的内容


# ---------------------------------------------------------------------------
# 2. LLM 节点：内部用 invoke，配合 messages 模式即可逐 token 产出
# ---------------------------------------------------------------------------
_llm = make_llm()

_SYSTEM_PROMPT = "你是一个幽默的段子手，回答要简短有趣。"


def call_model(state: State) -> dict:
    """调用 LLM 生成一段话，返回其文本内容。"""
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
    builder.add_node("call_model", call_model)

    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# 4. messages 模式：逐 token 输出
# ---------------------------------------------------------------------------
def main():
    graph = build_graph()

    print("stream_mode=\"messages\" —— 逐 token 流式输出：")
    for chunk, metadata in graph.stream(
        {"topic": "程序员"}, stream_mode="messages"
    ):
        # chunk 是 AIMessageChunk，text 部分用 .content 读取
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")

    # 演示一下 metadata 里有什么（只打印最后一条的元数据字段）
    print("metadata 字段示例：", list(metadata.keys()))

    # 打印流程图
    render_graphviz(graph, output_name="graph_stream_messages_test")


if __name__ == "__main__":
    main()