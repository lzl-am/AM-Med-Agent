from typing import Literal, Union
from anthropic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import Field


class RouteQuery(BaseModel):
    """
    定义数据模型
    question_type字段的值只能是 medication_guidance 或 other
    """

    question_type: Literal["medication_guidance", "other"] = Field(
        ...,
        description="给定用户问题，选择将其路由到medication_guidance或other",
    )


class QuestionRouter:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        """
        初始化语言模型和路由逻辑
        """
        # 初始化语言模型
        self.llm = ChatOllama(model=model_name)
        # 添加结构化输出
        # 将 llm 模型与 RouteQuestionType 数据模型关联起来
        self.structured_llm_router = self.llm.with_structured_output(RouteQuery)
        # Prompt
        self.system = """您是将用户问题路由到用药指导或其他问题的专家。\n\n
                如果你认为该问题是关于用药指导的，请输出 medication_guidance\n\n
                如果你认为该问题是其他类型的问题，请输出 other\n\n
                """
        self.route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system),
                ("human", "{question}"),
            ]
        )
        # 将 route_prompt 和 structured_llm_router 结合起来创建的
        # 根据用户的问题决定将问题路由到不同 Agent
        self.question_router = self.route_prompt | self.structured_llm_router

    def route_question(self, question: str) -> Union[dict, BaseModel]:
        """
        根据用户的问题决定将问题路由到不同的 Agent
        """
        return self.question_router.invoke({"question": question})


if __name__ == "__main__":
    question_router = QuestionRouter()
    print(question_router.route_question("发烧了，吃哪种药比较好？"))
    print(question_router.route_question("水是剧毒的吗？"))
