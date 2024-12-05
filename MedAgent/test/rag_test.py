import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from data_process.drug_instruction import getChineseMedicineInstructions


instruction_data_path = "../../data/medical_ner_entities.json"
instruction_list = getChineseMedicineInstructions(instruction_data_path)
# 将字符串列表转换为Document对象的列表
documents = [Document(page_content=instruction) for instruction in instruction_list]


model_path = os.path.abspath("../../embedding/sentence-transformer")
# 初始化一个HuggingFaceEmbedding对象，用于将文本转换为向量表示
embed_model = HuggingFaceEmbeddings(
    # 指定了一个预训练的sentence-transformer模型的路径
    model_name=model_path
)


# 使用Chroma类创建一个向量存储
# Chroma一般直接作为内存数据库使用，但是也可以进行持久化存储
vectorstore = Chroma.from_documents(
    documents=documents,
    collection_name="rag-chroma",
    embedding=embed_model,
)
# 创建一个检索器，用于检索文档
retriever = vectorstore.as_retriever()


if __name__ == '__main__':

    model = ChatOllama(model="qwen2.5:7b")
    resp01 = model.invoke("最近有点胃胀气，吃哪些中药进行调养比较好？")
    print(resp01.content)

    print("------------------------")

    documents01 = retriever.invoke(resp01.content)
    print(documents01)

    print("=========================")

    resp02 = model.invoke("帮我提取这段话涉及的药物列表：" + resp01.content)
    print(resp02.content)

    print("------------------------")

    documents02 = retriever.invoke(resp02.content)
    print(documents02)
