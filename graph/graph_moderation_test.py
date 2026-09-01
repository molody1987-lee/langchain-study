"""基于 LangGraph 的内容审核系统示例。

演示一个「多级人工审批」的内容审核流程：
    1. 自动打分：用 DeepSeek 结构化输出产出三个分数（毒性 / 垃圾 / 质量），写入 State。
    2. 依分数路由：
       - 严重违规（毒性或垃圾 >= 0.8）→ 直接三级终审
       - 存疑（毒性/垃圾 >= 0.5，或质量 < 0.5）→ 一级初审
       - 否则 → 自动通过
    3. 多级审批（human-in-the-loop）：
       - 每一级都用 interrupt() 暂停图，等待人工用 Command(resume=...) 批复。
       - 批复动作支持 approve（通过）/ reject（拒绝）/ escalate（升级到下一级）。
       - 一级 → 二级 → 三级（终审）逐级升级，全程留痕 audit_log。

关键 API：
    - interrupt(payload)                          节点内暂停，payload 会呈现给人工
    - Command(resume={"action": "..."})            人工批复后继续执行
    - checkpointer（MemorySaver）                  interrupt 暂停与续跑的前提
    - add_conditional_edges + 路由函数             按分数 / 决策分发下一节点

运行（项目根目录）：
    python -m graph.graph_moderation_test
"""

import operator
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from graph.graph_utils import render_graphviz
from llm.deepseek.global_setting import make_llm


# ---------------------------------------------------------------------------
# 1. 状态定义：内容 + 三个分数 + 审批结果与留痕
# ---------------------------------------------------------------------------
class State(BaseModel):
    content: str = ""                                  # 待审核内容
    toxicity_score: float = 0.0                        # 毒性分数（0~1，越高越有毒）
    spam_score: float = 0.0                            # 垃圾/广告分数（0~1）
    quality_score: float = 0.0                         # 质量分数（0~1，越高越好）
    reason: str = ""                                   # 打分说明
    decision: str = ""                                 # 最近一次人工/自动决策
    audit_log: Annotated[list[str], operator.add] = []  # 审核留痕（reducer 累加）
    result: str = ""                                   # 最终结论


# ---------------------------------------------------------------------------
# 2. 打分节点：DeepSeek 结构化输出
# ---------------------------------------------------------------------------
class ModerationScores(BaseModel):
    """DeepSeek 打分结果：三个分数 + 理由。"""

    toxicity_score: float = Field(ge=0, le=1, description="毒性分数 0~1，越高表示攻击/仇恨/暴力等有害内容越多")
    spam_score: float = Field(ge=0, le=1, description="垃圾/广告分数 0~1，越高表示越像垃圾营销内容")
    quality_score: float = Field(ge=0, le=1, description="内容质量分数 0~1，越高表示内容质量越好")
    reason: str = Field(description="一句话说明打分依据")


# DeepSeek 需用 function_calling 方式做结构化输出（不支持 json_schema response_format）
_score_llm = make_llm().with_structured_output(ModerationScores, method="function_calling")

_SCORE_SYSTEM_PROMPT = (
    "你是内容审核打分助手。请对用户给出的内容，从毒性（toxicity）、垃圾（spam）、"
    "质量（quality）三个维度打分（均为 0~1 的小数），并说明理由。"
)


def score_content(state: State) -> dict:
    """打分节点：调用 DeepSeek 结构化输出，产出三个分数。"""
    result = _score_llm.invoke(
        [SystemMessage(content=_SCORE_SYSTEM_PROMPT), HumanMessage(content=state.content)]
    )
    return {
        "toxicity_score": result.toxicity_score,
        "spam_score": result.spam_score,
        "quality_score": result.quality_score,
        "reason": result.reason,
        "audit_log": [f"DeepSeek 打分：{result.reason}"],
    }


# ---------------------------------------------------------------------------
# 3. 多级审批节点：每一级用 interrupt 暂停，等待人工批复
# ---------------------------------------------------------------------------
def _ask_human(level: str, state: State) -> str:
    """暂停图，把内容与分数呈现给人工，返回其批复动作。"""
    decision = interrupt(
        {
            "level": level,
            "content": state.content,
            "scores": {
                "toxicity": state.toxicity_score,
                "spam": state.spam_score,
                "quality": state.quality_score,
            },
            "note": "请批复：approve=通过 / reject=拒绝 / escalate=升级到下一级",
        }
    )
    return decision.get("action", "reject")


def level1_review(state: State) -> dict:
    action = _ask_human("一级初审", state)
    return {"decision": action, "audit_log": [f"一级初审：{action}"]}


def level2_review(state: State) -> dict:
    action = _ask_human("二级复审", state)
    return {"decision": action, "audit_log": [f"二级复审：{action}"]}


def level3_review(state: State) -> dict:
    action = _ask_human("三级终审", state)
    return {"decision": action, "audit_log": [f"三级终审：{action}"]}


# ---------------------------------------------------------------------------
# 4. 路由函数：按分数或人工决策决定下一节点
# ---------------------------------------------------------------------------
def route_by_scores(state: State) -> str:
    """打分后路由：严重违规直接终审，存疑一级初审，否则自动通过。"""
    if state.toxicity_score >= 0.8 or state.spam_score >= 0.8:
        return "level3_review"
    if state.toxicity_score >= 0.5 or state.spam_score >= 0.5 or state.quality_score < 0.5:
        return "level1_review"
    return "approve"


def route_after_level1(state: State) -> str:
    if state.decision == "approve":
        return "approve"
    if state.decision == "escalate":
        return "level2_review"
    return "reject"


def route_after_level2(state: State) -> str:
    if state.decision == "approve":
        return "approve"
    if state.decision == "escalate":
        return "level3_review"
    return "reject"


def route_after_level3(state: State) -> str:
    """三级是终审，不允许再升级。"""
    return "approve" if state.decision == "approve" else "reject"


# ---------------------------------------------------------------------------
# 5. 终止节点
# ---------------------------------------------------------------------------
def node_approve(state: State) -> dict:
    return {"result": "审核通过：内容允许发布", "audit_log": ["结论：通过"]}


def node_reject(state: State) -> dict:
    return {"result": "审核拒绝：内容被拦截", "audit_log": ["结论：拒绝"]}


# ---------------------------------------------------------------------------
# 6. 建图
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(State)
    builder.add_node("score_content", score_content)
    builder.add_node("level1_review", level1_review)
    builder.add_node("level2_review", level2_review)
    builder.add_node("level3_review", level3_review)
    builder.add_node("approve", node_approve)
    builder.add_node("reject", node_reject)

    builder.add_edge(START, "score_content")
    builder.add_conditional_edges(
        "score_content",
        route_by_scores,
        {"approve": "approve", "level1_review": "level1_review", "level3_review": "level3_review"},
    )
    builder.add_conditional_edges(
        "level1_review",
        route_after_level1,
        {"approve": "approve", "reject": "reject", "level2_review": "level2_review"},
    )
    builder.add_conditional_edges(
        "level2_review",
        route_after_level2,
        {"approve": "approve", "reject": "reject", "level3_review": "level3_review"},
    )
    builder.add_conditional_edges(
        "level3_review",
        route_after_level3,
        {"approve": "approve", "reject": "reject"},
    )
    builder.add_edge("approve", END)
    builder.add_edge("reject", END)

    # checkpointer 是 interrupt 暂停与续跑的前提
    return builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 7. 运行：对每条内容，在每次暂停时按 decisions 顺序给出人工决策
# ---------------------------------------------------------------------------
def run_case(graph, config, content, decisions):
    print(f"\n内容：{content!r}")
    graph.invoke({"content": content}, config)

    i = 0
    snapshot = graph.get_state(config)
    while snapshot.next:
        print(f"  [暂停] 等待节点 {snapshot.next} 的人工决策…")
        action = decisions[i] if i < len(decisions) else "reject"
        i += 1
        print(f"  [人工] 批复 -> {action}")
        graph.invoke(Command(resume={"action": action}), config)
        snapshot = graph.get_state(config)

    v = snapshot.values
    print(f"  分数：毒性={v['toxicity_score']:.2f} 垃圾={v['spam_score']:.2f} 质量={v['quality_score']:.2f}")
    print(f"  理由：{v['reason']}")
    print(f"  结论：{v['result']}")
    print(f"  留痕：{' | '.join(v['audit_log'])}")


def main():
    graph = build_graph()

    # 每条内容附带一个「人工决策序列」：审核节点每次 interrupt 暂停时依次取一个。
    # 分数由 DeepSeek 给出，具体走几级 / 是否自动通过取决于打分结果；
    # 决策里的 escalate / reject / approve 用来演示各级批复与升级动作。
    cases = [
        ("正常内容", "今天天气真好，下午一起去爬山", ["approve"]),
        ("辱骂内容-一级拒绝", "你真是个笨蛋", ["reject"]),
        ("疑似违规-逐级升级", "你这个垃圾，太蠢了", ["escalate", "escalate", "approve"]),
        ("暴力内容-终审拒绝", "我要杀了你，去死吧", ["reject"]),
        ("营销内容-升级复审", "加微信低价代购，点击领取优惠券", ["escalate", "reject"]),
    ]

    print("=" * 66)
    print("内容审核系统：多级人工审批（human-in-the-loop，DeepSeek 打分）")
    print("=" * 66)
    for i, (title, content, decisions) in enumerate(cases):
        print("\n" + "-" * 66)
        print(f"用例 {i + 1}：{title}")
        print("-" * 66)
        run_case(graph, {"configurable": {"thread_id": f"mod-{i}"}}, content, decisions)

    render_graphviz(graph, output_name="graph_moderation_test")


if __name__ == "__main__":
    main()