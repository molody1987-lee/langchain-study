import os
from typing import Optional

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated

from llm.deepseek import global_setting

llm = ChatOpenAI(model=global_setting.MODEL,base_url=os.environ["OPENAI_BASE_URL"])
# TypedDict
class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]

    # Alternatively, we could have specified setup as:

    # setup: str                    # no default, no description
    # setup: Annotated[str, ...]    # no default, no description
    # setup: Annotated[str, "foo"]  # default, no description

    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]


structured_llm = llm.with_structured_output(Joke, method="function_calling")

result=structured_llm.invoke("Tell me a joke about cats")
print(result)

