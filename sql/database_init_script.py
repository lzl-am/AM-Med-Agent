import os
import pandas as pd
import mysql.connector
from dotenv import load_dotenv


load_dotenv()


# MySQL数据库连接配置
db_config = {
    'host': os.getenv("MYSQL_HOST"),          # MySQL服务器地址
    'user': os.getenv("MYSQL_USER"),          # 数据库用户名
    'password': os.getenv("MYSQL_PASSWORD"),  # 数据库密码
    'database': os.getenv("MYSQL_DATABASE")   # 数据库名称
}

# 连接到MySQL数据库
try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # 读取CSV文件到DataFrame
    csv_file = '../data_process/medicine_instructions.csv'  # CSV文件路径
    df = pd.read_csv(csv_file, encoding='utf-8')

    # 用 NULL 替换 NaN（如果数据库字段支持 NULL）
    df = df.where(pd.notnull(df), None)

    # SQL 插入语句
    insert_query = """
    INSERT INTO drug_information (
        product_name, drug_name, approval_number, composition, dosage_form, 
        specification, efficacy, dosage_and_usage, adverse_reactions, 
        precautions, interactions, therapeutic_effect, packaging, 
        manufacturer, category, indication
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    # 将DataFrame中的数据转换为列表
    data_to_insert = []
    for index, row in df.iterrows():
        data = (
            row['药品商品名称'], row['药品名称'], row['批准文号'], row['成分'], row['剂型'],
            row['规格'], row['功效'], row['用法用量'], row['不良反应'], row['注意事项'],
            row['相互作用'], row['疗效'], row['药品包装'], row['制药公司'], row['43'], row['功能主治']
        )
        data_to_insert.append(data)

    print(data_to_insert)

    # 批量插入数据
    try:
        cursor.executemany(insert_query, data_to_insert)
        conn.commit()
        print(f"Successfully inserted {len(data_to_insert)} rows into the database.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        conn.rollback()


except mysql.connector.Error as err:
    print(f"Error: {err}")
finally:
    # 关闭数据库连接
    if cursor:
        cursor.close()
    if conn:
        conn.close()
