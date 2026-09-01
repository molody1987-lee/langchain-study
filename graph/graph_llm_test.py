"""基于 LLM 的路由决策示例（LangGraph）。

核心：路由节点不写死 if/else，而是让 LLM 用「结构化输出」的方式判断请求该交给哪个
处理节点，并附上决策理由。DeepSeek 需用 with_structured_output(method="function_calling")
（不支持 json_schema response_format），随后用条件边按 LLM 的决策结果分发。

流程：
    START → route(LLM 决策) → 条件边分发 → tech / billing / aftersales / general → END
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from graph.graph_utils import render_graphviz
from llm.deepseek.global_setting import make_llm


# ---------------------------------------------------------------------------
# 1. 状态 与 LLM 结构化决策模型
# ---------------------------------------------------------------------------
class State(BaseModel):
    query: str = ""    # 用户原始输入
    route: str = ""    # LLM 决策出的路由目标
    reason: str = ""   # LLM 给出的决策理由
    result: str = ""   # 对应处理节点的产出


class RouteDecision(BaseModel):
    """LLM 路由决策的结构化输出：目标部门 + 决策理由。"""

    target: Literal["tech", "billing", "aftersales", "general"] = Field(
        description="最合适的处理部门：tech 技术 / billing 账单 / aftersales 售后 / general 其它"
    )
    reason: str = Field(description="一句话说明为什么这样分类")


# LLM 需以可解析（结构化）的形式返回决策；DeepSeek 兼容方式为 function_calling
_router_llm = make_llm().with_structured_output(RouteDecision, method="function_calling")

_SYSTEM_PROMPT = "你是客服工单分流助手，判断用户问题该由哪个部门处理。"


# ---------------------------------------------------------------------------
# 2. 节点：路由决策 + 各处理分支
# ---------------------------------------------------------------------------
def node_route(state: State) -> dict:
    """路由节点：由 LLM 判断走哪个分支。"""
    decision = _router_llm.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=state.query)]
    )
    return {"route": decision.target, "reason": decision.reason}


def node_tech(state: State) -> dict:
    return {"result": f"[技术支持] 已转接：{state.query}"}


def node_billing(state: State) -> dict:
    return {"result": f"[账单] 已转接：{state.query}"}


def node_aftersales(state: State) -> dict:
    return {"result": f"[售后] 已转接：{state.query}"}


def node_general(state: State) -> dict:
    return {"result": f"[通用] 已转接：{state.query}"}


# ---------------------------------------------------------------------------
# 3. 建图：条件边按 LLM 决策的 route 分发
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(State)
    builder.add_node("route", node_route)
    builder.add_node("tech", node_tech)
    builder.add_node("billing", node_billing)
    builder.add_node("aftersales", node_aftersales)
    builder.add_node("general", node_general)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        lambda s: s.route,  # 返回 LLM 决策出的 route 作为分支键
        {
            "tech": "tech",
            "billing": "billing",
            "aftersales": "aftersales",
            "general": "general",
        },
    )
    for name in ["tech", "billing", "aftersales", "general"]:
        builder.add_edge(name, END)

    return builder.compile()


# ---------------------------------------------------------------------------
# 4. 运行示例
# ---------------------------------------------------------------------------
def main():
    graph = build_graph()
    queries = [
        "我的订单付款后一直没有显示成功，钱已经扣了",
        "家里宽带连不上，路由器指示灯一直闪红灯",
        "买的鞋子尺码不合适想退货，怎么操作？",
        "推荐一部好看的电影吧",
    ]

    for q in queries:
        result = graph.invoke({"query": q})
        print(f"用户问题：{q}")
        print(f"  路由决策 → {result['route']}（理由：{result['reason']}）")
        print(f"  处理结果 → {result['result']}")
        print("=" * 60)

    # 渲染流程图
    render_graphviz(graph, output_name="graph_llm_test")


if __name__ == "__main__":
    main()