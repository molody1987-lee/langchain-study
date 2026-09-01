"""
LangGraph 状态管理与持久化示例。

演示两个核心概念：
    1. 状态（State）管理
       - 用 Annotated + reducer（operator.add）让字段「累加」，而不是「覆盖」。
       - 用 checkpointer 让状态在节点边界持久化，同一 thread_id 可多次 invoke 续跑。
       - 用 update_state / get_state / get_state_history 手动读写检查点状态。
    2. 两种 checkpointer 的持久化区别
       - MemorySaver：内存版，进程内共享；新建实例即丢失（适合临时/测试）。
       - SqliteSaver：磁盘版，写入 sqlite 文件；新建实例/重启进程后状态仍在（生产推荐）。

运行：
    python graph_status_test.py
"""

import operator
import os
import sqlite3
from typing import Annotated

from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


# ---------------------------------------------------------------------------
# 1. 定义状态：演示「reducer 累加」与「普通覆盖」两种字段
# ---------------------------------------------------------------------------
class State(BaseModel):
    # reducer 字段：每次节点返回 {"counter": 1} 时，是「加 1」而非「赋值 1」
    counter: Annotated[int, operator.add] = 0
    # reducer 字段：list 用 operator.add 表示「追加」，历史会一直累积
    history: Annotated[list[str], operator.add] = []
    # 普通字段：无 reducer，节点返回即整体覆盖，保留最后一次写入
    label: str = ""


# ---------------------------------------------------------------------------
# 2. 定义节点
# ---------------------------------------------------------------------------
def node_step(state: State) -> dict:
    """演示：counter 累加、history 追加、label 覆盖。"""
    return {"counter": 1, "history": ["step"], "label": "processing"}


def node_done(state: State) -> dict:
    """收尾节点：继续追加 history、覆盖 label。"""
    return {"history": ["done"], "label": "completed"}


# ---------------------------------------------------------------------------
# 3. 构建图：checkpointer 从外部注入，方便复用同一图结构测试两种实现
# ---------------------------------------------------------------------------
def build_graph(checkpointer):
    builder = StateGraph(State)

    builder.add_node("step", node_step)
    builder.add_node("done", node_done)

    builder.add_edge(START, "step")
    builder.add_edge("step", "done")
    builder.add_edge("done", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 4. 演示一：MemorySaver（内存，不跨实例持久）
# ---------------------------------------------------------------------------
def demo_memory_saver():
    print("=" * 66)
    print("MemorySaver：内存 checkpointer")
    print("=" * 66)

    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "mem-1"}}

    # 第一次运行
    r1 = graph.invoke({"history": ["init"]}, config)
    print("第一次 invoke:")
    print("  counter =", r1["counter"], "| label =", r1["label"])
    print("  history =", r1["history"])

    # 同线程第二次运行（带新输入）—— reducer 会「累加」到已有状态之上
    r2 = graph.invoke({"history": ["turn-2"]}, config)
    print("第二次 invoke（同 thread_id，reducer 累加）:")
    print("  counter =", r2["counter"], "| label =", r2["label"])
    print("  history =", r2["history"])

    # 同线程 invoke(None)：从最近检查点「续跑」（当前已到 END，结果不变 / 幂等）
    r3 = graph.invoke(None, config)
    print("invoke(None) 续跑（已在 END，幂等）:")
    print("  counter =", r3["counter"], "| history 长度 =", len(r3["history"]))

    # 关键点：新建一个 MemorySaver 实例（模拟进程重启），旧状态读不到了
    graph_new = build_graph(MemorySaver())
    snap = graph_new.get_state(config)
    print("新建 MemorySaver 实例后 get_state（内存不持久，应为空）:")
    print("  values =", dict(snap.values), "| next =", snap.next)


# ---------------------------------------------------------------------------
# 5. 演示二：SqliteSaver（磁盘，跨实例/进程持久）
# ---------------------------------------------------------------------------
def demo_sqlite_saver():
    print()
    print("=" * 66)
    print("SqliteSaver：磁盘持久化 checkpointer")
    print("=" * 66)

    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_status_test.sqlite")
    # 删除旧库，保证每次运行输出确定（否则重复运行会在同一 thread_id 上持续累加）
    if os.path.exists(db_path):
        os.remove(db_path)
    config = {"configurable": {"thread_id": "sql-1"}}

    # 第一次运行（写库）
    graph1 = build_graph(SqliteSaver(sqlite3.connect(db_path, check_same_thread=False)))
    r1 = graph1.invoke({"history": ["init"]}, config)
    print("第一次 invoke 结果:", dict(r1))

    # 关键点：新建一个「全新图 + 全新 SqliteSaver」，指向同一 sqlite 文件
    # —— 模拟进程重启后，状态依然能读出来（磁盘持久化）。
    graph2 = build_graph(
        SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))
    )
    snap = graph2.get_state(config)
    print("新建实例（模拟重启）后 get_state:")
    print("  counter =", snap.values["counter"], "| label =", snap.values["label"])
    print("  history =", snap.values["history"])

    # get_state_history：按「新 -> 旧」列出每个节点边界保存过的检查点
    print("get_state_history（新 -> 旧）:")
    for i, item in enumerate(graph2.get_state_history(config)):
        print(
            f"  [{i}] counter={item.values.get('counter')} "
            f"label={item.values.get('label')} next={item.next}"
        )

    # update_state：手动改写检查点状态（普通字段直接覆盖）
    graph2.update_state(config, {"label": "manually_updated"})
    after = graph2.get_state(config)
    print("update_state 手动改写 label 后:")
    print("  label =", after.values["label"], "| counter（未动） =", after.values["counter"])


# ---------------------------------------------------------------------------
# 6. 演示三：多线程隔离 —— 同一 checkpointer，不同 thread_id 互不干扰
# ---------------------------------------------------------------------------
def demo_multi_thread():
    print()
    print("=" * 66)
    print("多线程隔离：同图 + 同 checkpointer，不同 thread_id 各自独立")
    print("=" * 66)

    # 用内存 sqlite，既是「持久化 checkpointer」，又保证每次运行结果确定、不产生额外文件
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    graph = build_graph(SqliteSaver(conn))

    cfg_a = {"configurable": {"thread_id": "user-A"}}
    cfg_b = {"configurable": {"thread_id": "user-B"}}

    # 线程 A 跑两次，线程 B 只跑一次
    graph.invoke({"history": ["A-第1次"]}, cfg_a)
    graph.invoke({"history": ["A-第2次"]}, cfg_a)
    graph.invoke({"history": ["B-第1次"]}, cfg_b)

    a = graph.get_state(cfg_a)
    b = graph.get_state(cfg_b)

    print("线程 A（user-A，invoke 两次）:")
    print("  counter =", a.values["counter"])
    print("  history =", a.values["history"])
    print("线程 B（user-B，invoke 一次）:")
    print("  counter =", b.values["counter"])
    print("  history =", b.values["history"])

    print("=> 两个会话的 counter / history 各自独立累加，互不污染")

    # 各自的检查点历史也只包含本线程，彼此不可见
    n_a = sum(1 for _ in graph.get_state_history(cfg_a))
    n_b = sum(1 for _ in graph.get_state_history(cfg_b))
    print(f"=> 各线程快照历史：user-A 有 {n_a} 个，user-B 有 {n_b} 个（互不相见）")


def main():
    demo_memory_saver()
    demo_sqlite_saver()
    demo_multi_thread()


if __name__ == "__main__":
    main()