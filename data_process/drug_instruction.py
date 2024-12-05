import csv
import json
import re


def getChineseMedicineInstructions(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    items = []
    for item in data:
        items.append(item['text'])

    return items


def extractInfo(text):
    # 正则表达式匹配所有【】内的内容
    pattern = r"【(.*?)】(.*?)\s*(?=【|$)"
    # 使用re.findall找到所有匹配项
    matches = re.findall(pattern, text)
    # 将结果存入字典
    info_dict = {key.strip(): value.strip() for key, value in matches}
    return info_dict


def save_to_csv(data_list, filename):
    if not data_list:
        print("数据列表为空，无法保存到CSV文件！")
        return

    # 过滤掉药品名称和药品商品名称都为空或仅含空格的数据，并填充为空的字段
    data_list = [data for data in data_list if data.get("药品商品名称", "").strip() or data.get("药品名称", "").strip()]

    for data in data_list:
        # 去除空格后的判断
        # 如果药品商品名称为空且药品名称不为空，则用药品名称填充药品商品名称
        if not data.get("药品商品名称", "").strip() and data.get("药品名称", "").strip():
            data["药品商品名称"] = data["药品名称"]

        # 如果药品名称为空且药品商品名称不为空，则用药品商品名称填充药品名称
        if not data.get("药品名称", "").strip() and data.get("药品商品名称", "").strip():
            data["药品名称"] = data["药品商品名称"]

    # 确保列表非空，若为空则返回
    if not data_list:
        print("所有数据均为空或无效，未保存任何数据。")
        return

    # 提取data_list[0]中的字段名
    fieldnames = data_list[0].keys()

    # 确保所有字典的字段名一致
    for data in data_list:
        # 删除任何不在fieldnames中的额外字段
        for key in list(data.keys()):
            if key not in fieldnames:
                del data[key]

    # 打开CSV文件进行写入
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()

        # 写入每条数据
        for data in data_list:
            writer.writerow(data)

    print(f"数据已保存到 {filename} 文件中！")


if __name__ == '__main__':
    instructions = getChineseMedicineInstructions('../data/medical_ner_entities.json')
    print(len(instructions))
    print(instructions[:10])

    dataList = [extractInfo(text) for text in instructions]
    print(len(dataList))
    print(dataList[:3])

    # 保存到CSV文件
    save_to_csv(dataList, 'medicine_instructions.csv')
