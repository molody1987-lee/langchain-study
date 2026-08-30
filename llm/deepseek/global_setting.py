import logging
import os
import sys

from dotenv import load_dotenv

# 从项目根目录的 .env 文件加载环境变量（DASHSCOPE_API_KEY 等）
load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL="https://api.deepseek.com"
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY", "")
LANGCHAIN_TRACING_V2="true"
MODEL="deepseek-chat"

# 通义千问 DashScope（用于 Embedding，DeepSeek 本身不提供向量化接口）
# API Key 从 .env 读取，不再硬编码在源码中
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDING_MODEL = "qwen3.7-text-embedding"
# 向量维度：None=模型默认(1024)；可选 2560/2048/1536/1024/768/512/256
# 改维度后需删除 milvus.db 重建向量库
EMBEDDING_DIMENSION = 1024

# Milvus 向量库本地文件路径（Milvus Lite，免 Docker，数据落盘到本地）
MILVUS_URI = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "milvus.db",
)


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

from langchain_core.globals import set_debug, set_verbose
from langchain_openai import ChatOpenAI

set_verbose(True)  # 打印每步输入/输出
set_debug(True)


def make_llm() -> ChatOpenAI:
    """创建已配置好的 ChatOpenAI（DeepSeek）实例。"""
    return ChatOpenAI(
        model=MODEL,
        base_url=os.environ["OPENAI_BASE_URL"],
        verbose=True,
    )

# 1. 配置基础日志输出（格式、级别）
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # 确保输出到标准输出流
)

# 2. 显式开启 LangChain 内部核心日志
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)

# 如果需要查看API调用的详细请求和响应（注意可能会包含敏感信息）
# logging.getLogger("httpx").setLevel(logging.INFO)

# 3. 你的 LangChain 代码...
print("日志配置已完成，开始执行...")
# from langchain.llms import OpenAI
# llm = OpenAI()
# llm("你好")


