import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


llm = ChatOpenAI(
    model="internlm2.5-latest",
    api_key=os.getenv("INTERNLM_API_KEY"),
    base_url="https://internlm-chat.intern-ai.org.cn/puyu/api/v1/"
)

print(llm.invoke("水是剧毒的吗？"))
print(llm.invoke("我一直拉稀，请给我一些药品推荐"))
