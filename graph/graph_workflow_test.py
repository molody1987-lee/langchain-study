# -*- coding: utf-8 -*-
"""
基于 LangGraph 的图流程示例，演示三类能力：
    1. 节点失败自动重试（RetryPolicy）
    2. 重试耗尽后，修复后从失败节点续跑（checkpointer + invoke(None)）
    3. 人工介入（interrupt 暂停 + Command(resume) 批复）
同时使用 graphviz 包渲染流程图。

依赖安装：
    pip install langgraph graphviz
    # 渲染成 PNG/SVG 图片还需要系统安装 Graphviz（提供 dot 命令）
    # macOS: brew install graphviz
    # 提示：未安装 graphviz 包时会自动回退为 LangGraph 内置的 mermaid 源码。
"""

import os
import sqlite3
from typing import Literal

from pydantic import BaseModel

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt

from graph.graph_utils import render_graphviz


# ---------------------------------------------------------------------------
# 1. 定义状态（State）：节点之间传递的数据结构
# ---------------------------------------------------------------------------
class State(BaseModel):
    text: str = ""       # 原始输入文本
    category: str = ""   # 分类结果
    result: str = ""     # 最终处理结果


# ---------------------------------------------------------------------------
# 2. 定义节点函数：每个节点接收 state，返回要更新的部分字段
# ---------------------------------------------------------------------------
def node_generate(state: State) -> dict:
    """生成文本。"""
    return {"text": state.text + " -> generated"}


def node_classify(state: State) -> dict:
    """对文本进行分类：含 sensitive 走人工审核，含 ok 为 positive，否则 negative。"""
    text = state.text
    if "sensitive" in text:
        return {"category": "sensitive"}
    return {"category": "positive" if "ok" in text else "negative"}


def node_handle_positive(state: State) -> dict:
    """正向分支处理。"""
    return {"result": f"[positive] processed: {state.text}"}


def node_handle_negative(state: State) -> dict:
    """负向分支处理。"""
    return {"result": f"[negative] processed: {state.text}"}


_RISKY_CALLS = 0   # 模块级计数器：记录 risky 节点被调用的次数（仅用于模拟不稳定）


def node_risky(state: State) -> dict:
    """模拟不稳定节点：前 FAIL_TIMES 次调用抛异常，之后成功。

    - 配合 RetryPolicy：自动重试，直到成功或重试次数耗尽。
    - 重试耗尽后异常向外抛出；修复条件后用 invoke(None) 从该节点续跑。

    注意：这里抛 ConnectionError，因为 RetryPolicy 默认只重试「可重试」类异常
    （如连接错误、5xx 等）；RuntimeError/ValueError 等编程错误默认不会重试。
    """
    global _RISKY_CALLS
    fail_times = int(os.environ.get("FAIL_TIMES", "0"))
    _RISKY_CALLS += 1
    if _RISKY_CALLS <= fail_times:
        raise ConnectionError(f"[risky] 第 {_RISKY_CALLS} 次调用失败（模拟网络不稳定）")
    return {"text": state.text + " -> risk_ok"}


def node_human_review(state: State) -> dict:
    """人工审批节点：用 interrupt 暂停图，等待人工用 Command(resume=...) 批复。"""
    decision = interrupt(
        {
            "text": state.text,
            "question": "检测到敏感内容，请人工裁定：approve 通过 / reject 拒绝",
        }
    )
    verdict = decision.get("verdict", "reject")
    return {"result": f"[人工审批] 结论={verdict}；文本={state.text}"}


def route_by_category(state: State) -> Literal["handle_positive", "handle_negative", "human_review"]:
    """条件路由：根据 category 决定下一个节点。"""
    category = state.category
    if category == "positive":
        return "handle_positive"
    if category == "negative":
        return "handle_negative"
    return "human_review"


# ---------------------------------------------------------------------------
# 3. 构建图：添加节点、边、条件边
# ---------------------------------------------------------------------------
def build_graph(checkpointer=None):
    builder = StateGraph(State)

    builder.add_node("generate", node_generate)
    # risky 节点：模拟不稳定服务，用 RetryPolicy 自动重试
    builder.add_node(
        "risky",
        node_risky,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.3, backoff_factor=2.0),
    )
    builder.add_node("classify", node_classify)
    builder.add_node("handle_positive", node_handle_positive)
    builder.add_node("handle_negative", node_handle_negative)
    builder.add_node("human_review", node_human_review)  # interrupt 暂停等人工

    builder.add_edge(START, "generate")                 # 开始 -> generate
    builder.add_edge("generate", "risky")               # generate -> risky（可重试）
    builder.add_edge("risky", "classify")               # risky -> classify
    builder.add_conditional_edges(                      # classify -> (条件分支)
        "classify",
        route_by_category,
        {
            "handle_positive": "handle_positive",
            "handle_negative": "handle_negative",
            "human_review": "human_review",
        },
    )
    builder.add_edge("handle_positive", END)            # handle_positive -> 结束
    builder.add_edge("handle_negative", END)            # handle_negative -> 结束
    builder.add_edge("human_review", END)               # human_review -> 结束

    # checkpointer 使图在节点边界持久化，是「续跑」和「interrupt 暂停」的前提
    return builder.compile(checkpointer=checkpointer)


def _reset_risky(fail_times: int = 0):
    """重置不稳定节点：清空调用计数并设置故障次数（0 表示不故障）。"""
    global _RISKY_CALLS
    _RISKY_CALLS = 0
    os.environ["FAIL_TIMES"] = str(fail_times)


def run_normal_case(graph, config, text):
    _reset_risky()
    result = graph.invoke({"text": text}, config)
    print(f"  输入: {text}")
    print(f"  结果: {result.get('result')}")


def run_auto_retry_case(graph, config, text, fail_times=2):
    """前 fail_times 次失败，之后成功 —— 演示 RetryPolicy 自动重试。"""
    _reset_risky(fail_times)
    result = graph.invoke({"text": text}, config)
    print(f"  FAIL_TIMES={fail_times}（自动重试后成功）")
    print(f"  结果: {result.get('result')}")


def run_resume_case(graph, config, text):
    """重试耗尽抛出异常 -> 人工修复 -> invoke(None) 从失败节点续跑。"""
    _reset_risky(99)  # 远超 max_attempts=3，模拟持续失败
    print("  触发持续失败…")
    try:
        graph.invoke({"text": text}, config)
    except Exception as exc:  # noqa: BLE001
        print(f"  [异常] 重试已耗尽: {exc}")
        print("  [人工] 模拟修复外部依赖…")
        _reset_risky(0)  # 修复后清除故障条件
        result = graph.invoke(None, config)  # None = 从 checkpoint 续跑（重跑失败节点）
        print(f"  [续跑] 从失败节点恢复后结果: {result.get('result')}")


def run_hitl_case(graph, config, text, verdict="approve"):
    """敏感内容 -> interrupt 暂停 -> 人工 Command(resume=...) 批复后继续。"""
    _reset_risky()
    print(f"  输入: {text}")
    graph.invoke({"text": text}, config)  # 在 human_review 的 interrupt 处暂停
    snapshot = graph.get_state(config)
    while snapshot.next:
        print(f"  [暂停] 等待人工裁定…（模拟选择: {verdict}）")
        graph.invoke(Command(resume={"verdict": verdict}), config)
        snapshot = graph.get_state(config)
    print(f"  结果: {snapshot.values.get('result')}")


def main():
    conn = sqlite3.connect("graph_wf_test.sqlite", check_same_thread=False)
    graph = build_graph(checkpointer=SqliteSaver(conn))

    print("=" * 60)
    print("1) 正常执行（无故障）")
    run_normal_case(graph, {"configurable": {"thread_id": "t1"}}, "hello ok")

    print("=" * 60)
    print("2) 自动重试：失败 2 次后成功")
    run_auto_retry_case(graph, {"configurable": {"thread_id": "t2"}}, "hello ok", fail_times=2)

    print("=" * 60)
    print("3) 重试耗尽 -> 修复 -> 从失败节点续跑")
    run_resume_case(graph, {"configurable": {"thread_id": "t3"}}, "hello ok")

    print("=" * 60)
    print("4) 人工介入：interrupt + Command(resume)")
    run_hitl_case(graph, {"configurable": {"thread_id": "t4"}}, "hello sensitive content")

    print("=" * 60)
    print("5) 流程图渲染")
    render_graphviz(graph, output_name="graph_test", output_format="png")


if __name__ == "__main__":
    main()