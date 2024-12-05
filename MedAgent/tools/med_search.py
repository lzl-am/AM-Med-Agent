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


class PubMedInput(BaseModel):
    """Input for the PubMed tool."""

    query: str = Field(description="search query to look up")


class PubMedSearchResults(BaseTool):
    """
    医学文献查询，并返回查询结果
    https://cpubmed.openi.org.cn/graphwiki/apiI
    """
    name: str = "pubmed_search"
    description: str = (
        "一个医学搜索引擎，用于搜索医学文献。"
        "当你需要回答与医学有关的问题时非常有用"
        "输入应该是一个搜索查询。"
    )
    args_schema: Type[BaseModel] = PubMedInput
    max_results: int = 5
    api_key: str = os.getenv("CPUBMED_KG_API_KEY")

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> list:
        """Use the tool."""
        url = f'http://cpubmed.openi.org.cn/graph/retrieve?query={query}&sign={self.api_key}'
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            data = response.json()
            return data[:self.max_results]
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise HTTPException(status_code=400, detail="Request to PubMed API failed")
        except ValueError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise HTTPException(status_code=400, detail="Failed to parse JSON response from PubMed API")


if __name__ == "__main__":
    pubmed_search = PubMedSearchResults()
    print(pubmed_search.invoke("我最近有点咳嗽，吃点什么药好"))
