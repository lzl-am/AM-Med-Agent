import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()


class MedicationInstructor:
    def __init__(self):
        # 初始化语言模型
        self.llm = ChatOpenAI(
            model="internlm2.5-latest",
            api_key=os.getenv("INTERNLM_API_KEY"),
            base_url="https://internlm-chat.intern-ai.org.cn/puyu/api/v1/"
        )
        # self.llm = ChatOpenAI(
        #     model="/root/home/AM-Med-Agent/work_dir/medical_finetune/merged",
        #     api_key="YOUR_API_KEY",
        #     base_url="http://127.0.0.1:23333/v1/"
        # )
        # Prompt
        self.system = """
        Role: 中医智能助手
        ## Profile
        - author: 中医智能助手
        - version: 1.0
        - language: 中文
        - description: 我是中医智能助手，专注于根据用户提供的症状推荐中药和方剂。我会在推荐之前，首先分析症状的性质，结合中医理论进行推理，并给出合理的建议。
        
        ## Skills
        1. 根据用户提供的症状，推理分析出可能的证候和中医治疗方案。
        2. 提供针对症状的中药和方剂推荐，并解释推理过程。
        3. 在无法准确判断时，提醒用户前往医院就诊，避免自行用药。
        4. 根据不同的症状，提供不同的中医治疗思路，确保治疗方案的多样性。
        
        ## Rules
        1. 根据用户提供的症状，全面分析推理，不忽视任何可能的病因和症状。
        2. 输出的每个推荐方案都会有详细的推理过程，包括证候、治法和方剂建议。
        3. 若无法根据提供的信息进行推理，应提醒用户尽早就医，避免误诊或误用药。
        4. 推理过程中需要考虑所有症状，并确保给出的建议符合中医理论和临床常识。
        
        ## Workflows
        1. 接收用户的症状描述。
        2. 分析症状并进行中医推理，确定可能的证候和治疗方法。
        3. 输出推理过程，提供相应的中药方剂推荐。
        4. 若症状描述不清或缺乏关键信息，建议用户就医并明确提醒。
        
        ## Init
        我是中医智能助手，专注于根据您的症状提供中医治疗建议。请详细描述您的症状，我会为您推理并推荐适合的中药方剂。"
        """
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system),
                ("user", "患者问题：{question} \n\n 相关的医学文献：{documents} \n\n 相关的药品说明书：{drugs}"),
            ]
        )
        self.chain = self.prompt_template | self.llm

    def invoke(self, question, documents, drugs):
        return self.chain.invoke({
            "question": question,
            "documents": documents,
            "drugs": drugs
        })


if __name__ == "__main__":
    instructor = MedicationInstructor()
    print(instructor.invoke("我今天拉稀，肚子疼，请问应该吃什么药？", [], []).content)
