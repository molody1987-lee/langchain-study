"""基于 LangGraph 的「Agent 按需调用 Skill」示例。

Skill = 一段可复用的专业指令/知识，打包成 name + 简介 + 完整内容。
采用「渐进式披露」（progressive disclosure）策略：

    1. 系统提示里只放每个 Skill 的「name + 简介」——一个轻量目录；
    2. Agent 判断需要某个 Skill 时，调用 load_skill 工具「按需加载」其完整内容；
    3. 加载后的完整指令以 ToolMessage 进入上下文，Agent 据此继续完成任务。

这样即便有几十上百个 Skill，也不会一次性撑爆上下文窗口。

流程（ReAct 循环）：
    START → agent(LLM 决定是否调用工具) ⇄ 条件边(是否还有工具调用) → END

运行（项目根目录）：
    python -m graph.graph_skill_test
"""

from datetime import datetime
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel, Field

from graph.graph_utils import render_graphviz
from llm.deepseek.global_setting import make_llm


# ---------------------------------------------------------------------------
# 1. 定义 Skill 目录：只有 name + description 会进入系统提示（轻量）
# ---------------------------------------------------------------------------
class Skill(BaseModel):
    name: str
    description: str = Field(description="一句话简介，用于让 Agent 判断是否需要用这个技能")
    content: str = Field(description="完整指令/知识，按需加载")


SKILLS = [
    Skill(
        name="return_policy",
        description="公司退货/退款政策（退货条件、运费谁出、退款到账时间等）",
        content=(
            "【退货政策】\n"
            "1. 订单签收后 7 天内可申请退货，商品需保持原样、不影响二次销售。\n"
            "2. 非质量问题的退货运费由客户承担；质量问题的退货运费由公司承担。\n"
            "3. 退款在仓库签收退货后 3-5 个工作日内原路退回。\n"
            "4. 定制类、生鲜类商品不支持无理由退货。"
        ),
    ),
    Skill(
        name="tech_support",
        description="产品技术支持与常见故障排查步骤",
        content=(
            "【技术支持排查流程】\n"
            "1. 确认设备是否正常通电，检查指示灯状态。\n"
            "2. 尝试重启设备，等待 30 秒后再开机。\n"
            "3. 若仍未解决，引导用户记录错误码并升级人工客服。\n"
            "4. 常见错误码：E01=网络异常，E02=传感器故障。"
        ),
    ),
]

_SKILL_BY_NAME = {s.name: s for s in SKILLS}


# ---------------------------------------------------------------------------
# 2. 工具：load_skill 按需加载技能，另配一个普通工具做对照
# ---------------------------------------------------------------------------
@tool
def load_skill(name: str) -> str:
    """当你需要某个具体技能（退货政策 return_policy / 技术支持 tech_support）的详细内容时调用。

    Args:
        name: 技能名称。
    """
    skill = _SKILL_BY_NAME.get(name)
    if skill is None:
        return f"未找到技能 [{name}]，可用技能：{list(_SKILL_BY_NAME)}"
    return skill.content


@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [load_skill, get_current_time]
_TOOL_BY_NAME = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# 3. 系统提示：只放 Skill 目录（name + 简介），不提前塞完整内容
# ---------------------------------------------------------------------------
def build_system_prompt() -> str:
    catalog = "\n".join(f"- {s.name}: {s.description}" for s in SKILLS)
    return (
        "你是某公司的智能客服助手，负责回答退货政策、技术支持等问题。\n\n"
        "你可以按需使用以下技能（Skill）。初始只看到简介；"
        "当你需要某个技能的详细内容时，请调用 load_skill 工具加载它：\n"
        f"{catalog}\n\n"
        "请基于加载到的完整指令准确作答；与技能无关的问题直接回答即可。"
    )


# ---------------------------------------------------------------------------
# 4. State 与 Agent（ReAct 循环，复用 DeepSeek）
# ---------------------------------------------------------------------------
class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration: int = 0


_llm = make_llm().bind_tools(TOOLS)
MAX_STEPS = 6  # 防死循环


def agent(state: State) -> dict:
    response = _llm.invoke([SystemMessage(content=build_system_prompt())] + state.messages)

    out = [response]
    # 若 LLM 决定调用工具（可能 load_skill，也可能是 get_current_time），执行并回填结果
    for tc in response.tool_calls:
        result = _TOOL_BY_NAME[tc["name"]].invoke(tc["args"])
        out.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return {"messages": out, "iteration": state.iteration + 1}


def should_continue(state: State) -> Literal["agent", "end"]:
    for m in reversed(state.messages):
        if isinstance(m, AIMessage):
            if m.tool_calls and state.iteration < MAX_STEPS:
                return "agent"
            return "end"
    return "end"


def build_graph():
    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"agent": "agent", "end": END})
    return builder.compile()


# ---------------------------------------------------------------------------
# 5. 运行与结果展示
# ---------------------------------------------------------------------------
def _final_answer(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            return m.content
    return "(无回答)"


def _loaded_skills(messages) -> list[str]:
    """从消息里挑出本次实际调用过 load_skill 的技能名（用于展示）。"""
    loaded = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                if tc["name"] == "load_skill":
                    loaded.append(tc.get("args", {}).get("name", "?"))
    return loaded


def main():
    graph = build_graph()

    queries = [
        "我想退货，请问运费由谁承担？",
        "我的设备一直显示 E01 错误码，该怎么办？",
        "现在几点了？",
    ]

    print("=" * 66)
    print("Agent 按需调用 Skill 示例")
    print("=" * 66)

    for i, q in enumerate(queries, 1):
        print(f"\n【问题 {i}】{q}")
        result = graph.invoke({"messages": [HumanMessage(content=q)]})
        loaded = _loaded_skills(result["messages"])
        print(f"  加载的技能：{loaded if loaded else '（未加载，直接回答）'}")
        print(f"  最终回答：{_final_answer(result['messages'])}")

    render_graphviz(graph, output_name="graph_skill_test")


if __name__ == "__main__":
    main()