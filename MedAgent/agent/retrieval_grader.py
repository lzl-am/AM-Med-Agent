import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from MedAgent.tools.med_search import PubMedSearchResults

load_dotenv()


class GradeRetrieval(BaseModel):
    """
    用于表示一个二进制分数，用于检查检索到的文档的相关性
    """
    # 存储文档是否与问题相关的二进制分数（yes 或 no）
    binary_score: Literal['yes', 'no'] = Field(
        description="内容是否与问题相关，返回'yes'或'no'",
    )


class RetrievalGrader:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        # 初始化语言模型
        self.llm = ChatOllama(model=model_name)
        # self.llm = ChatOpenAI(
        #     model="internlm2.5-latest",
        #     api_key=os.getenv("INTERNLM_API_KEY"),
        #     base_url="https://internlm-chat.intern-ai.org.cn/puyu/api/v1/"
        # )

        self.structured_llm_grader = self.llm.with_structured_output(GradeRetrieval)
        self.system = """你是一名专业的审核员，负责评估检索到的文档与用户问题的相关性。
        如果你认为文档内容与问题相关，请返回'yes'，否则返回'no'。"""
        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader

    def grade(self, question: str, document: str) -> GradeRetrieval:
        return self.retrieval_grader.invoke({"question": question, "document": document})


if __name__ == "__main__":
    pubmed_search = PubMedSearchResults()
    resp = pubmed_search.invoke("我最近有点咳嗽，吃点什么药好")
    print(resp)
    retrieval_grader = RetrievalGrader()
    for doc in resp:
        print(str(doc))
        print(retrieval_grader.grade("我最近有点咳嗽，吃点什么药好", str(doc)))

