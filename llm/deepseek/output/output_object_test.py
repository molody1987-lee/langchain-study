import os
from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from llm.deepseek import global_setting

llm = ChatOpenAI(model=global_setting.MODEL,base_url=os.environ["OPENAI_BASE_URL"])
# Pydantic
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


structured_llm = llm.with_structured_output(Joke, method="function_calling")
result=structured_llm.invoke("Tell me a joke about cats")
print(result)

