from langchain_core.messages import HumanMessage

from llm.deepseek.global_setting import make_llm
from llm.deepseek.message.manual_with_history import TrimMessages, build_graph

if __name__ == "__main__":
    llm = make_llm()
    # 复用抽象：默认 max_tokens=500、token_counter="approximate"，即按 token 限制消息大小
    trimmer = TrimMessages(max_tokens=10, token_counter="approximate")
    graph = build_graph(llm, trimmer, "manual_with_history_token.sqlite")

    config = {"configurable": {"thread_id": "user-1"}}

    resp1 = graph.invoke({"messages": [HumanMessage("我叫小明")]}, config)
    print(resp1["messages"][-1].content)

    resp2 = graph.invoke({"messages": [HumanMessage("我叫什么？")]}, config)
    print(resp2["messages"][-1].content)