# 初始化Embedding类
import os.path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

model_path = os.path.abspath("../../embedding/sentence-transformer")

# 初始化一个HuggingFaceEmbedding对象，用于将文本转换为向量表示
embed_model = HuggingFaceEmbedding(
    # 指定了一个预训练的sentence-transformer模型的路径
    model_name=model_path
)


if __name__ == "__main__":
    # Embed文本
    embeddings = embed_model.get_text_embedding_batch(
        [
            "您好，有什么需要帮忙的吗？",
            "哦，你好！昨天我订的花几天送达",
            "请您提供一些订单号？",
            "12345678",
        ]
    )
    print(len(embeddings), len(embeddings[0]))

    # Embed查询
    embedded_query = embed_model.get_query_embedding("刚才对话中的订单号是多少?")
    print(embedded_query[:3])
