from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


load_dotenv()


class DiseaseEntity(BaseModel):
    """
    定义数据模型
    """
    disease: str = Field(description="疾病名称")


class DiseaseRouter:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        # 初始化语言模型
        self.llm = ChatOllama(model=model_name)
        # 添加结构化输出
        # 将 llm 模型与 RouteQuery 数据模型关联起来
        self.structured_llm_router = self.llm.with_structured_output(DiseaseEntity)
        # Prompt
        self.system = """你是一个专业的医学助手，你可以根据患者描述，判断患者所患疾病。疾病名称应该尽可能精确。"""
        self.route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system),
                ("human", "根据用户问题描述，提取疾病关键字，不要输出冗余信息：{question}"),
            ]
        )
        self.disease_router = self.route_prompt | self.structured_llm_router

    def invoke(self, question: str):
        return self.disease_router.invoke({"question": question})


if __name__ == "__main__":
    disease_router = DiseaseRouter()
    print(disease_router.invoke("我最近有点咳嗽"))
    print(disease_router.invoke("我最近有点咳嗽，该吃些什么药？"))
    print(disease_router.invoke("最近总是感觉头晕，应该怎么办？"))
