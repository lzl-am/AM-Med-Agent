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


def searchByName(keyword):
    # 连接到MySQL数据库
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 构建模糊查询的SQL语句
        query = """
        SELECT product_name, drug_name, composition, dosage_form, specification, 
        efficacy, dosage_and_usage, adverse_reactions, precautions, 
        interactions, therapeutic_effect, packaging, category, indication 
        FROM drug_information 
        WHERE product_name LIKE %s OR drug_name LIKE %s
        """

        # 执行模糊查询，%s 是占位符，防止SQL注入
        like_pattern = f"%{keyword}%"
        cursor.execute(query, (like_pattern, like_pattern))

        # 获取列名
        columns = [col[0] for col in cursor.description]

        # 获取查询结果
        results = cursor.fetchall()
        if results:
            # 将每一行结果转换为字典格式
            dict_results = [dict(zip(columns, row)) for row in results]
            return dict_results
        else:
            return None

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        # 关闭数据库连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def searchByEfficacy(keyword):
    # 连接到MySQL数据库
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 构建模糊查询的SQL语句
        query = """
        SELECT product_name, drug_name, composition, dosage_form, specification, 
        efficacy, dosage_and_usage, adverse_reactions, precautions, 
        interactions, therapeutic_effect, packaging, category, indication
        FROM drug_information 
        WHERE efficacy LIKE %s
        """

        # 执行模糊查询，%s 是占位符，防止SQL注入
        like_pattern = f"%{keyword}%"  # 在关键词前后添加通配符
        cursor.execute(query, (like_pattern,))

        # 获取列名
        columns = [col[0] for col in cursor.description]

        # 获取查询结果
        results = cursor.fetchall()
        if results:
            # 将每一行结果转换为字典格式
            dict_results = [dict(zip(columns, row)) for row in results]
            return dict_results
        else:
            return None

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        # 关闭数据库连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    drugs01 = searchByName("乌鸡")
    print(drugs01)
    drugs02 = searchByEfficacy("咳嗽")
    print(drugs02)
