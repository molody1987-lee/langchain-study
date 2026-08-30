import os

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

from llm.deepseek import global_setting

llm = ChatOpenAI(model=global_setting.MODEL,base_url=os.environ["OPENAI_BASE_URL"],verbose=True)

template='你是一个{role},请用{style}风格回答问题:{question}'

#prompt_template=PromptTemplate.from_template(template)
# filled_prompt=prompt_template.format_prompt(role='数学老师',style='通俗易懂',question='勾股定理是什么')
#filled_prompt=prompt_template.invoke({'role':'数学老师','style':'通俗易懂','question':'勾股定理是什么'})
sys_template='你是一个数学老师，请以{style}的风格回答问题'
user_template='请用简单易懂的方式解释：{question}'
prompt_template=ChatPromptTemplate.from_messages([('system',sys_template),('human',user_template)])
prompt_template=prompt_template.format_messages(style='生动有趣',question='勾股定理是什么？')
# for chunk in model.stream(filled_prompt):
#     print(chunk.content,end='',flush=True)

result=llm.invoke(prompt_template)
print(result.content)