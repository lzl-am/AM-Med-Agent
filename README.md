# AM-Med-Agent  

本项目依托：[InternLM](https://github.com/InternLM/Tutorial)  

## 项目背景

随着人工智能技术的发展，大模型在医疗领域的应用日益增多，尤其是在辅助诊断方面。为了确保这些模型能够提供精确的诊断结果，必须在模型投入使用之前进行周全的知识准备和数据库构建工作，以确保模型具备充足的信息资源。

## 项目目标

本项目旨在开发一个医疗辅助诊断 Agent，该 Agent 能够接收并分析患者的文本和影像数据，提供精确的诊断结果，并确保诊断过程的安全性和可靠性。

## 项目核心功能

1. **多模态数据接收与处理**
- 数据输入：系统将接收包括但不限于医疗记录、实验室报告、病理报告以及医学影像等多模态输入。
- 数据整合：将不同来源和类型的数据整合，形成统一的数据格式，以便模型进行分析。
2. **深入分析与知识检索**
- 知识检索：从构建的医疗知识数据库中提取与患者病情相关联的信息，以辅助诊断。
3. **生成增强技术**
- 精确性提升：通过生成增强技术，进一步提升诊断的精确性和洞察力。
- 结果优化：优化诊断结果，使其更加符合患者的实际病情。
4. **安全性与可靠性监控**
- 监控系统：建立一个监控系统，密切监视模型的输出。
- 响应识别：识别任何不确定或可能有害的响应。
- 警报机制：一旦发现问题，立即向医生或相关医疗专家发出警报。
- 审查与干预：提示医疗专家对模型的输出进行审查或必要时进行干预。

## 项目实施步骤

- 数据库构建：构建一个全面的医疗知识数据库，包括最新的医学研究、临床指南和病例数据。
- 模型训练：使用医疗数据对大模型进行训练，确保其能够理解和处理医疗领域的复杂数据。
- 系统开发：开发辅助诊断系统，集成多模态数据处理、知识检索和生成增强技术。
- 监控系统部署：部署监控系统，确保模型输出的安全性和可靠性。
- 测试与优化：在实际医疗场景中测试系统，收集反馈并进行优化。


## 数据集

- [ShenNong_TCM_Dataset](https://huggingface.co/datasets/michaelwzhu/ShenNong_TCM_Dataset)
- [中文医疗对话数据集](https://tianchi.aliyun.com/dataset/90163)
- [面向家庭常见疾病的知识图谱](http://data.openkg.cn/dataset/medicalgraph#)
- [中药说明书实体识别](https://tianchi.aliyun.com/dataset/86819)