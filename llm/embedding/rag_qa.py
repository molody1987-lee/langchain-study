"""完整的 RAG 检索问答流程。

流程：文档 → 切分 → 向量化(Embedding) → 向量库 → 检索 → 拼接上下文 → LLM 生成答案。

- Embedding 使用通义千问 DashScope（DeepSeek 本身不提供向量化接口）
- LLM 使用 DeepSeek
- 向量库使用 Milvus Lite 持久化存储（数据落盘到本地文件，进程重启后仍可检索）
"""

from dashscope import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm.deepseek import global_setting
from llm.deepseek.global_setting import make_llm

# 各模型单次支持的最大批量文本数（避免超限报错）
_BATCH_SIZE = {
    "qwen3.7-text-embedding": 20,
    "text-embedding-v1": 25,
    "text-embedding-v2": 25,
    "text-embedding-v3": 10,
    "text-embedding-v4": 10,
}


class DashScopeEmbeddings(Embeddings):
    """直接基于 DashScope SDK 实现的文本向量化，避免依赖已弃用的 langchain-community。"""

    def __init__(self, model: str, api_key: str, dimension: int | None = None):
        self.model = model
        self.api_key = api_key
        self.dimension = dimension

    def _embed(self, texts: list[str], text_type: str) -> list[list[float]]:
        batch_size = _BATCH_SIZE.get(self.model, 10)
        result: list[list[float]] = []
        # 分批调用，规避模型单次输入上限
        for i in range(0, len(texts), batch_size):
            resp = TextEmbedding.call(
                model=self.model,
                input=texts[i : i + batch_size],
                api_key=self.api_key,
                text_type=text_type,
                dimension=self.dimension,
            )
            if resp.status_code == 200:
                result += [item["embedding"] for item in resp.output["embeddings"]]
            else:
                raise ValueError(
                    f"DashScope Embedding 调用失败：{resp.code} {resp.message}"
                )
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档（索引阶段）。"""
        return self._embed(texts, text_type="document")

    def embed_query(self, text: str) -> list[float]:
        """向量化查询（检索阶段）。"""
        return self._embed([text], text_type="query")[0]


def make_embeddings() -> DashScopeEmbeddings:
    """创建 DashScope Embedding 实例（从环境变量读取 DASHSCOPE_API_KEY）。"""
    return DashScopeEmbeddings(
        model=global_setting.EMBEDDING_MODEL,
        api_key=global_setting.DASHSCOPE_API_KEY,
        dimension=global_setting.EMBEDDING_DIMENSION,
    )


def split_documents(docs: list[Document]) -> list[Document]:
    """把长文档按字符切分成更小的块，便于检索。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,      # 每块最多 200 字符
        chunk_overlap=20,    # 相邻块重叠 20 字符，避免语义被切断
    )
    return splitter.split_documents(docs)


def build_vectorstore(docs: list[Document], embeddings) -> Milvus:
    """把切分后的文档向量化并持久化到 Milvus，幂等写入（drop_old 重建同名 collection）。"""
    store = Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="langchain",
        connection_args={"uri": global_setting.MILVUS_URI},
        drop_old=True,   # 覆盖旧的同名 collection，避免重复入库
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