import sqlite3
from typing import Annotated, Any

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph, add_messages
from pydantic import BaseModel, Field

from llm.deepseek.global_setting import make_llm

SYSTEM_MESSAGE = SystemMessage(content="你是一个乐于助人的助手。")


# 用 Pydantic 定义图状态（替代 MessagesState，避免类型检查器报 StateLike 不匹配）
class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]


# 用 RunnableSerializable 封装裁剪逻辑，使其成为可序列化、可复用的一个步骤
class TrimMessages(RunnableSerializable[list[BaseMessage], list[BaseMessage]]):
    max_tokens: int = Field(
        default=500,
        description="保留的最大 token 数（当 token_counter=len 时表示最大消息条数）",
    )
    token_counter: Any = Field(
        default="approximate",
        description="token 计数方式：'approximate'（估算）或 len（按条数）",
    )

    def invoke(self, messages, config=None, **kwargs):
        return trim_messages(
            messages,
            max_tokens=self.max_tokens,
            strategy="last",           # 丢弃旧的、保留最近的
            token_counter=self.token_counter,
            include_system=True,       # 保留开头的 system 消息
            start_on="human",          # 裁剪后确保从 human 消息开始
        )


def make_call_model(llm: ChatOpenAI, trimmer: TrimMessages):
    """根据指定的 llm 与 trimmer 生成 call_model 节点函数，便于复用。"""

    def call_model(state: State) -> dict:
        # system 消息每次放在最前，再按上限裁剪历史，最后调用模型
        messages = trimmer.invoke([SYSTEM_MESSAGE] + state.messages)
        response = llm.invoke(messages)
        return {"messages": [response]}

    return call_model


def build_graph(llm: ChatOpenAI, trimmer: TrimMessages, db_path: str):
    """根据 llm、trimmer 与数据库路径构建带 SQLite 检查点的图。"""
    builder = StateGraph(State)
    builder.add_node("model", make_call_model(llm, trimmer))
    builder.add_edge(START, "model")

    # 用 SQLite 持久化历史，重启进程后仍能延续同一 thread_id 的对话
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return builder.compile(checkpointer=SqliteSaver(conn))


if __name__ == "__main__":
    llm = make_llm()
    trimmer = TrimMessages()
    graph = build_graph(llm, trimmer, "manual_with_history.sqlite")

    config = {"configurable": {"thread_id": "user-1"}}

    resp1 = graph.invoke({"messages": [HumanMessage("我叫小明")]}, config)
    print(resp1["messages"][-1].content)

    resp2 = graph.invoke({"messages": [HumanMessage("我叫什么？")]}, config)
    print(resp2["messages"][-1].content)