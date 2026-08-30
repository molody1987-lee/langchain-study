import sqlite3
from typing import Annotated

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph, add_messages
from pydantic import BaseModel

from llm.deepseek.global_setting import make_llm

SYSTEM_MESSAGE = SystemMessage(content="你是一个乐于助人的助手。")

SUMMARY_THRESHOLD = 6   # 消息数超过它就触发压缩
KEEP_RECENT = 4         # 压缩时只压缩更早的消息，保留最近 KEEP_RECENT 条


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str = ""    # 存放压缩后的历史摘要


def summarize(llm: ChatOpenAI, summary: str, to_compress: list) -> str:
    """让 LLM 把旧消息合并进已有摘要，输出新摘要。"""
    history = "\n".join(f"{m.__class__.__name__}: {m.content}" for m in to_compress)
    prompt = (
        "请把下面的对话历史提炼成简洁摘要，保留重要事实与决定，供后续对话使用。\n\n"
        f"【已有摘要】\n{summary or '（无）'}\n\n"
        f"【新增对话】\n{history}\n\n"
        "输出合并后的摘要（只输出摘要正文，不要解释）："
    )
    return llm.invoke(prompt).content


def make_call_model(llm: ChatOpenAI):
    def call_model(state: State) -> dict:
        messages = state.messages
        summary = state.summary

        if len(messages) > SUMMARY_THRESHOLD:
            # 1. 把最旧的消息压缩进摘要
            to_compress = messages[: -KEEP_RECENT]
            summary = summarize(llm, summary, to_compress)

            # 2. 用「摘要 + 最近消息」生成回复
            system = SystemMessage(content=f"历史摘要：\n{summary}")
            recent = messages[-KEEP_RECENT:]
            response = llm.invoke([system] + recent)

            # 3. 删除已被压缩的旧消息，防止状态无限增长
            deletes = [RemoveMessage(id=m.id) for m in to_compress]
            return {"messages": deletes + [response], "summary": summary}

        # 历史不长：直接正常回复
        response = llm.invoke([SYSTEM_MESSAGE] + messages)
        return {"messages": [response]}

    return call_model


def build_graph(llm: ChatOpenAI, db_path: str):
    """构建带 SQLite 检查点和摘要压缩的图。"""
    builder = StateGraph(State)
    builder.add_node("model", make_call_model(llm))
    builder.add_edge(START, "model")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return builder.compile(checkpointer=SqliteSaver(conn))


if __name__ == "__main__":
    llm = make_llm()
    graph = build_graph(llm, "manual_with_history_summary.sqlite")

    config = {"configurable": {"thread_id": "user-1"}}

    resp1 = graph.invoke({"messages": [HumanMessage("我叫小明")]}, config)
    print(resp1["messages"][-1].content)

    resp2 = graph.invoke({"messages": [HumanMessage("我叫什么？")]}, config)
    print(resp2["messages"][-1].content)