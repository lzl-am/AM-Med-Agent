import json
from tqdm import tqdm


system_prompt = """Role: 中医智能助手
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


def process_line(line):
    # 解析 JSON 行
    data = json.loads(line)

    processed_data = {
        "conversation": []
    }

    processed_data["conversation"].append({
        "system": system_prompt,
        "input": data["query"],
        "output": data["response"]
    })

    # 将处理后的对象转回 JSON 字符串
    return json.dumps(processed_data, ensure_ascii=False)


if __name__ == "__main__":
    input_file = "../data/ChatMed_TCM-v0.2.json"
    output_file = "../data/finetune_data/medical_datasets.jsonl"

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:
        # 计算总行数用于进度条
        total_lines = sum(1 for _ in infile)
        infile.seek(0)  # 重置文件指针到开头

        # 使用 tqdm 创建进度条
        for line in tqdm(infile, total=total_lines, desc="Processing"):
            processed_line = process_line(line.strip())
            outfile.write(processed_line + '\n')

    print("完整数据已处理并保存到", output_file)
    print("数据行数:", total_lines)

    output_small_file = "../data/finetune_data/medical_small_datasets.jsonl"
    # 设置最大处理行数为 10000
    max_lines = 10000

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_small_file, 'w', encoding='utf-8') as outfile:
        # 计算总行数并输出
        total_lines = sum(1 for _ in infile)
        infile.seek(0)  # 重置文件指针到开头

        # 初始化计数器
        line_count = 0

        # 使用 tqdm 创建进度条
        for line in tqdm(infile, total=min(total_lines, max_lines), desc="Processing"):
            if line_count >= max_lines:  # 如果已经处理了最大行数，退出循环
                break
            processed_line = process_line(line.strip())
            outfile.write(processed_line + '\n')
            line_count += 1