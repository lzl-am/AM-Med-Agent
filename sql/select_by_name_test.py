import os
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

# 要查询的关键词
keyword = '乌鸡'

# 连接到MySQL数据库
try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # 构建模糊查询的SQL语句
    query = """
    SELECT * 
    FROM drug_information 
    WHERE product_name LIKE %s OR drug_name LIKE %s
    """

    # 执行模糊查询，%s 是占位符，防止SQL注入
    like_pattern = f"%{keyword}%"
    cursor.execute(query, (like_pattern, like_pattern))

    # 获取查询结果
    results = cursor.fetchall()

    if results:
        for row in results:
            print(row)  # 打印查询结果
    else:
        print("没有找到符合条件的记录。")

except mysql.connector.Error as err:
    print(f"Error: {err}")
finally:
    # 关闭数据库连接
    if cursor:
        cursor.close()
    if conn:
        conn.close()
