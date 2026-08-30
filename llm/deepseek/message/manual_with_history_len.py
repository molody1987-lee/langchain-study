from langchain_core.messages import HumanMessage

from llm.deepseek.global_setting import make_llm
from llm.deepseek.message.manual_with_history import TrimMessages, build_graph

if __name__ == "__main__":
    llm = make_llm()
    # 复用抽象：token_counter=len、max_tokens=10，即按消息条数限制消息大小
    trimmer = TrimMessages(token_counter=len)
    graph = build_graph(llm, trimmer, "manual_with_history_len.sqlite")

    config = {"configurable": {"thread_id": "user-1"}}

    resp1 = graph.invoke({"messages": [HumanMessage("我叫小明")]}, config)
    print(resp1["messages"][-1].content)

    resp2 = graph.invoke({"messages": [HumanMessage("我叫什么？")]}, config)
    print(resp2["messages"][-1].content)