import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

from llm.deepseek import global_setting

llm = ChatOpenAI(model=global_setting.MODEL,base_url=os.environ["OPENAI_BASE_URL"],verbose=True)

template='你是一个{role},请用{style}风格回答问题:{question}'

prompt_template=PromptTemplate.from_template(template)
#filled_prompt=prompt_template.invoke({'role':'数学老师','style':'通俗易懂','question':'勾股定理是什么'})
# for chunk in model.stream(filled_prompt):
#     print(chunk.content,end='',flush=True)

llm_chain=prompt_template | llm|StrOutputParser()
result=llm_chain.invoke({'role':'数学老师','style':'通俗易懂','question':'勾股定理是什么'})
print(result)