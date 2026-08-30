"""基于本地 bge-small-zh-v1.5 模型的完整 RAG 检索问答流程。

与 rag_qa.py 的区别：Embedding 使用本地 BGE 模型（无需联网 API、无需 API Key），
向量库使用 Chroma 持久化存储。

流程：文档 → 切分 → 向量化(本地 bge) → 向量库 → 检索 → 拼接上下文 → LLM 生成答案。
"""

import os

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm.deepseek.global_setting import make_llm

# 本地 BGE 模型名
LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# BGE 模型查询时的指令前缀（提升检索效果）
_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
# 本地向量库使用独立目录，避免与 rag_qa 的向量库冲突
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_CHROMA_DIR = os.path.join(_BASE_DIR, "chroma_local")


class BgeEmbeddings(Embeddings):
    """带查询指令前缀的 BGE 封装：query 加前缀，document 不加。"""

    def __init__(self, model_name: str):
        self._inner = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # 归一化，便于余弦相似度
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(_QUERY_INSTRUCTION + text)


def make_embeddings() -> BgeEmbeddings:
    """创建本地 BGE Embedding 实例（首次运行会自动下载模型权重）。"""
    return BgeEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)


def split_documents(docs: list[Document]) -> list[Document]:
    """把长文档按字符切分成更小的块，便于检索。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,      # 每块最多 200 字符
        chunk_overlap=20,    # 相邻块重叠 20 字符，避免语义被切断
    )
    return splitter.split_documents(docs)


def build_vectorstore(docs: list[Document], embeddings) -> Chroma:
    """把切分后的文档向量化并持久化到 Chroma，幂等写入（覆盖同名 collection）。"""
    # 先清理旧的 langchain_local collection，避免多次运行导致重复入库
    client = chromadb.PersistentClient(path=LOCAL_CHROMA_DIR)
    try:
        client.delete_collection("langchain_local")
    except Exception:
        pass

    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=LOCAL_CHROMA_DIR,
        collection_name="langchain_local",
    )
    return store


# 检索后拼接给 LLM 的提示词模板：限定模型只依据上下文回答
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个助手，请只根据下面提供的【上下文】回答问题。"
            "如果上下文中没有答案，就如实回答“根据现有资料无法回答”。\n\n"
            "【上下文】\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def answer(llm, question: str, retrieved: list[Document]) -> str:
    """把检索到的文档拼成上下文，交给 LLM 生成答案。"""
    context = "\n\n".join(doc.page_content for doc in retrieved)
    chain = PROMPT | llm
    return chain.invoke({"context": context, "question": question}).content


if __name__ == "__main__":
    # 1. 准备原始文档（模拟知识库）
    raw_docs = [
        Document(
            page_content="小明是一只柯基犬，今年三岁，喜欢吃苹果。它的主人叫李明，住在北京。",
        ),
        Document(
            page_content="苹果富含维生素和膳食纤维，有助于消化。柯基犬要控制糖分摄入，苹果要去核后少量喂食。",
        ),
        Document(
            page_content="北京今天天气晴朗，适合带狗去公园散步。李明经常在周末带小明去朝阳公园。",
        ),
    ]

    # 2. 切分 + 向量化 + 入库
    embeddings = make_embeddings()
    split_docs = split_documents(raw_docs)
    store = build_vectorstore(split_docs, embeddings)

    # 3. LLM
    llm = make_llm()

    # 4. 检索 + 生成（问答）
    question = "小明喜欢吃什么？"
    retrieved = store.similarity_search(question, k=2)
    print("检索到的片段：")
    for doc in retrieved:
        print(f"  - {doc.page_content}")

    result = answer(llm, question, retrieved)
    print("\n最终回答：")
    print(result)