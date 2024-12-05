import os
import requests
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("CPUBMED_KG_API_KEY")
entity = "咳嗽"

url = f'http://cpubmed.openi.org.cn/graph/schema?entity={entity}&sign={api_key}'

# 发送GET请求
response = requests.get(url)

# 检查请求是否成功
if response.status_code == 200:
    # 解析返回的JSON数据
    data = response.json()
    print(data)

    for disease, info in data.items():
        print(f"疾病：{disease}")
        print(info.keys())
        print(f"药物治疗：{info['药物治疗']}")

else:
    print(f'请求失败，状态码：{response.status_code}')
