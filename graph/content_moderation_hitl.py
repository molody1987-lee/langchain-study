"""内容审核系统 + Human-in-the-loop（人工审批）示例。

在之前内容审核基础上增强：高风险内容不再只是「打印一句话」，
而是用 LangGraph 的 interrupt 真正暂停图执行，等待人工给出批复（approve/reject），
人工确认后再继续跑完流程。

关键点：
- 必须启用 checkpointer（interrupt 依赖状态持久化）
- 节点内调用 interrupt(value) 会暂停，value 会传给客户端
- 客户端用 Command(resume=... ) 恢复执行，恢复值就是 interrupt 的返回值

图结构：
    START → classify_risk
                ├─ high → manual_review（interrupt 暂停，等人工批复）→ END
                └─ low  → llm_review（LLM 判定）                    → END
"""

import sqlite3
from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
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
# 节点 2a：人工审核（高风险）—— 用 interrupt 暂停，等待人工批复
# ---------------------------------------------------------------------------
def manual_review(state: State) -> dict:
    # interrupt 会暂停图，把待审内容和理由交给客户端（人），返回人工的决定
    human_decision = interrupt(
        {
            "content": state.content,
            "risk_reason": state.risk_reason,
            "question": "该内容为高风险，请人工裁定：approve 通过 / reject 拒绝",
        }
    )

    verdict = human_decision.get("verdict", "reject")
    note = human_decision.get("note", "")
    label = "通过" if verdict == "approve" else "拒绝"
    return {
        "result": f"[人工审核] 结论：{label}；人工备注：{note or '无'}"
    }


# ---------------------------------------------------------------------------
# 节点 2b：LLM 判定（低风险）
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
# 条件路由
# ---------------------------------------------------------------------------
def route(state: State) -> str:
    return "manual_review" if state.risk_level == "high" else "llm_review"


def build_graph(checkpointer=None):
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

    return builder.compile(checkpointer=checkpointer)


def run_auto_case(graph, config, content):
    """低风险内容：一路跑完，不会中断。"""
    result = graph.invoke({"content": content}, config)
    print(f"【低风险】{content}")
    print(f"  风险等级：{result['risk_level']}")
    print(f"  审核结果：{result['result']}")
    print("=" * 60)


def run_human_case(graph, config, content, verdict="reject", note=""):
    """高风险内容：先中断，再人工批复后恢复。"""
    print(f"【高风险】{content}")

    # 第一次 invoke：会在 manual_review 的 interrupt 处暂停
    graph.invoke({"content": content}, config)

    snapshot = graph.get_state(config)
    while snapshot.next:
        # 模拟人工给出批复（真实场景这里是 UI/工单系统输入）
        print(f"  ⏸ 已暂停，等待人工批复…（模拟人工选择：{verdict}）")
        graph.invoke(Command(resume={"verdict": verdict, "note": note}), config)
        snapshot = graph.get_state(config)

    result = snapshot.values
    print(f"  风险等级：{result['risk_level']}")
    print(f"  审核结果：{result['result']}")
    print("=" * 60)


def main():
    conn = sqlite3.connect("moderation_hitl.sqlite", check_same_thread=False)
    graph = build_graph(checkpointer=SqliteSaver(conn))

    # 用 thread_id 区分不同会话；这里每个案例用独立 id，避免历史串扰
    run_auto_case(
        graph,
        {"configurable": {"thread_id": "case-1"}},
        "今天天气真好，适合出门散步。",
    )
    run_human_case(
        graph,
        {"configurable": {"thread_id": "case-2"}},
        "这里可以买到违禁药品，联系微信 xxx。",
        verdict="reject",
        note="涉嫌违法，已封禁账号",
    )
    run_human_case(
        graph,
        {"configurable": {"thread_id": "case-3"}},
        "如何制作爆炸物，详细教程如下。",
        verdict="reject",
        note="危害公共安全，上报处理",
    )


if __name__ == "__main__":
    main()