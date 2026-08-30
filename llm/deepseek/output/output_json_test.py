import os

from langchain_openai import ChatOpenAI

from llm.deepseek import global_setting

llm = ChatOpenAI(model=global_setting.MODEL,base_url=os.environ["OPENAI_BASE_URL"])
json_schema = {
    "title": "joke",
    "description": "Joke to tell user.",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "The setup of the joke",
        },
        "punchline": {
            "type": "string",
            "description": "The punchline to the joke",
        },
        "rating": {
            "type": "integer",
            "description": "How funny the joke is, from 1 to 10",
            "default": None,
        },
    },
    "required": ["setup", "punchline"],
}
structured_llm = llm.with_structured_output(json_schema, method="function_calling")

result=structured_llm.invoke("Tell me a joke about cats")
print(result)