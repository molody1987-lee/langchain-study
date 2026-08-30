"""基于 Agent 的天气预报工具示例（带多轮对话记忆）。

使用 LangGraph 的 create_react_agent 构建一个能根据用户提问自主决定
是否调用「天气预报」工具的智能体，并通过 SqliteSaver 持久化对话历史，
让 Agent 能记住上一轮的上下文，支持多轮追问。

流程：用户提问 → Agent 判断是否需要工具 → 调用 get_weather 工具 →
拿到结果 → 组织成自然语言回答（记忆靠 checkpointer 按 thread_id 隔离）。
"""

import json
import sqlite3
import urllib.parse
import urllib.request

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent
from langchain_core.tools import tool

from llm.deepseek.global_setting import make_llm

# wttr.in 免费天气接口，无需 API Key
_WEATHER_URL = "https://wttr.in/{city}?format=j1"


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气情况。

    Args:
        city: 城市名称，可以是中文（如「北京」）或英文（如「Beijing」）。
    """
    url = _WEATHER_URL.format(city=urllib.parse.quote(city))
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"查询天气失败：{e}"

    try:
        current = data["current_condition"][0]
        desc = current["lang_zh"][0]["value"] if "lang_zh" in current else current["weatherDesc"][0]["value"]
        temp_c = current["temp_C"]
        humidity = current["humidity"]
        wind = current["windspeedKmph"]
        result = (
            f"{city}当前天气：{desc}；温度 {temp_c}℃；"
            f"湿度 {humidity}%；风速 {wind} km/h。"
        )
        return result
    except (KeyError, IndexError) as e:
        return f"解析天气数据失败：{e}"


def main():
    """创建带记忆的 Agent 并进行多轮对话测试。"""
    llm = make_llm()

    # 用 SqliteSaver 持久化对话历史，进程重启后记忆仍在
    conn = sqlite3.connect("tools_test.sqlite", check_same_thread=False)
    agent = create_agent(
        llm,
        tools=[get_weather],
        checkpointer=SqliteSaver(conn),
    )

    # 固定的 thread_id 保证同一会话内共享历史；换成别的 id 就是新会话
    config = {"configurable": {"thread_id": "weather-chat-1"}}

    # 第一轮：直接问天气
    questions = [
        "杭州今天天气怎么样？",
        "那适合穿什么衣服出门呢？",  # 追问，依赖上一轮天气结果，无需再传城市名
    ]

    for question in questions:
        print(f"提问：{question}\n")
        result = agent.invoke({"messages": [("user", question)]}, config=config)
        answer = result["messages"][-1].content
        print("回答：")
        print(answer)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()