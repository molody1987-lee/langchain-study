"""内容审核系统 Agent 示例。

流程：先由 LLM 对内容做风险分级，再根据风险等级分流——
    - 风险高（high）→ 走人工审核（转交人工，不直接由 LLM 放行）
    - 风险低（low） → 走 LLM 判定（通过 / 拒绝）

图结构：
    START → classify_risk（风险分级）
                ├─ 条件边: high → manual_review（人工审核）→ END
                └─ 条件边: low  → llm_review（LLM 判定）   → END
"""

from typing import Annotated, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from llm.deepseek.global_setting import make_llm


class State(BaseModel):
    content: str = ""           # 待审核内容
    risk_level: str = ""        # 风险等级：high / low
    risk_reason: str = ""       # 分级理由
    result: str = ""            # 最终审核结果


# ---------------------------------------------------------------------------
# 节点 1：风险分级
# ---------------------------------------------------------------------------
class RiskAssessment(BaseModel):
    risk_level: Literal["high", "low"] = Field(
        description="风险等级：high 表示涉及色情、暴力、政治敏感、诈骗等高风险，low 表示风险较低"
    )
    reason: str = Field(description="简要说明分级理由")


_classify_llm = make_llm().with_structured_output(RiskAssessment, method="function_calling")

_CLASSIFY_PROMPT = (
    "你是一个内容审核员。请判断以下内容的违规风险等级。\n"
    "若涉及色情、暴力、血腥、政治敏感、仇恨言论、诈骗、违法违规等内容，判定为 high；\n"
    "否则判定为 low。只输出等级和理由。\n\n"
)


def classify_risk(state: State) -> dict:
    assessment = _classify_llm.invoke(_CLASSIFY_PROMPT + state.content)
    return {"risk_level": assessment.risk_level, "risk_reason": assessment.reason}


# ---------------------------------------------------------------------------
# 节点 2a：人工审核（高风险走这里）
# ---------------------------------------------------------------------------
def manual_review(state: State) -> dict:
    return {
        "result": (
            f"[转人工审核] 内容被判定为高风险，已转交人工复核。\n"
            f"风险等级：{state.risk_level}；分级理由：{state.risk_reason}"
        )
    }


# ---------------------------------------------------------------------------
# 节点 2b：LLM 判定（低风险走这里）
# ---------------------------------------------------------------------------
class ReviewDecision(BaseModel):
    verdict: Literal["pass", "reject"] = Field(description="审核结论：pass 通过，reject 拒绝")
    reason: str = Field(description="判定理由")


_review_llm = make_llm().with_structured_output(ReviewDecision, method="function_calling")

_REVIEW_PROMPT = (
    "你是一个内容审核员。请对以下低风险内容做出最终判定："
    "无明显问题则 pass，存在轻微违规（如不当引导、低俗擦边、广告营销）则 reject。\n"
    "只输出结论和理由。\n\n"
)


def llm_review(state: State) -> dict:
    decision = _review_llm.invoke(_REVIEW_PROMPT + state.content)
    label = "通过" if decision.verdict == "pass" else "拒绝"
    return {"result": f"[LLM 判定] 结论：{label}；理由：{decision.reason}"}


# ---------------------------------------------------------------------------
# 条件路由：高风险转人工，低风险交给 LLM
# ---------------------------------------------------------------------------
def route(state: State) -> str:
    return "manual_review" if state.risk_level == "high" else "llm_review"


def build_graph():
    builder = StateGraph(State)
    builder.add_node("classify_risk", classify_risk)
    builder.add_node("manual_review", manual_review)
    builder.add_node("llm_review", llm_review)

    builder.add_edge(START, "classify_risk")
    builder.add_conditional_edges(
        "classify_risk",
        route,
        {"manual_review": "manual_review", "llm_review": "llm_review"},
    )
    builder.add_edge("manual_review", END)
    builder.add_edge("llm_review", END)

    return builder.compile()


def main():
    graph = build_graph()

    samples = [
        ("低风险示例", "今天天气真好，适合出门散步。"),
        ("低风险示例", "这款产品限时优惠，快来抢购吧！"),
        ("高风险示例", "这里可以买到违禁药品，联系微信 xxx。"),
        ("高风险示例", "如何制作爆炸物，详细教程如下。"),
    ]

    for title, content in samples:
        result = graph.invoke({"content": content})
        print(f"【{title}】{content}")
        print(f"  风险等级：{result['risk_level']}（{result['risk_reason']}）")
        print(f"  审核结果：{result['result']}")
        print("=" * 60)


if __name__ == "__main__":
    main()