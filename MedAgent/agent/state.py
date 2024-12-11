from typing import List
from typing_extensions import TypedDict


# Define the state for the agent
class State(TypedDict):
    question: str
    keyword: str
    documents: List[str]
    drugs: List[str]
    instruction: str
