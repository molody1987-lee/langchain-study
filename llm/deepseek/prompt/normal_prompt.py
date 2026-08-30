import os

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from llm.deepseek import global_setting

model = ChatOpenAI(model=global_setting.MODEL,base_url=os.environ["OPENAI_BASE_URL"],verbose=True)

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

parser = StrOutputParser()

chain=model | parser

chain.invoke(messages)