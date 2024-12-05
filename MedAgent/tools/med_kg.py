import os
from typing import Optional, Type
import requests
from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.logger import logger
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel


load_dotenv()


class PubMedKGInput(BaseModel):
    """Input for the PubMedKG tool."""

    entity: str = Field(description="search entity to look up")


class PubMedKGResults(BaseTool):
    """
    医学知识图谱查询，并返回查询结果
    https://cpubmed.openi.org.cn/graphwiki/apiI
    """
    name: str = "pubmed_kg_search"
    description: str = (
        "一个医学知识图谱，用于搜索与查询实体相关联的三元组。"
        "当你需要回答与医学有关的问题时非常有用。"
        "输入应该是一个疾病名称。"
    )
    args_schema: Type[BaseModel] = PubMedKGInput
    api_key: str = os.getenv("CPUBMED_KG_API_KEY")

    def _run(
        self,
        entity: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> list:
        """Use the tool."""
        url = f'http://cpubmed.openi.org.cn/graph/schema?entity={entity}&sign={self.api_key}'
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            data = response.json()
            return data
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise HTTPException(status_code=400, detail="Request to PubMed API failed")
        except ValueError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise HTTPException(status_code=400, detail="Failed to parse JSON response from PubMed API")


if __name__ == "__main__":
    pubmed_kg = PubMedKGResults()
    data = pubmed_kg.invoke("咳嗽")
    for disease, info in data.items():
        print(f"疾病：{disease}")
        print(info.keys())
        print(f"药物治疗：{info['药物治疗']}")
