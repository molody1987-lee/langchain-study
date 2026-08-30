# 1. 配置基础日志输出（格式、级别）
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # 确保输出到标准输出流
)

# 2. 显式开启 LangChain 内部核心日志
logging.getLogger("langchain").setLevel(logging.INFO)
# 如果需要查看API调用的详细请求和响应（注意可能会包含敏感信息）
# logging.getLogger("httpx").setLevel(logging.INFO)