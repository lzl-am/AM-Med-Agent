from langgraph.constants import START, END
from langgraph.graph import StateGraph
from MedAgent.agent.medication_instructor import MedicationInstructor
from MedAgent.agent.retrieval_grader import RetrievalGrader
from MedAgent.agent.state import State
from MedAgent.router.disease_router import DiseaseRouter
from MedAgent.tools.med_kg import PubMedKGResults
from MedAgent.tools.med_search import PubMedSearchResults
from MedAgent.tools.sql_executor import searchByName, searchByEfficacy

pubmed_search = PubMedSearchResults()
retrieval_grader = RetrievalGrader()
disease_router = DiseaseRouter()
pubmed_kg = PubMedKGResults()
med_instructor = MedicationInstructor()


def search_documents(state):
    question = state["question"]

    documents = pubmed_search.invoke(question)

    state["documents"] = documents

    return state


def documents_filter(state):
    documents = state["documents"]
    question = state["question"]

    res = []
    for document in documents:
        score = retrieval_grader.grade(question, document)
        grade = score.binary_score
        if grade == "yes":
            res.append(document["f_Abstract"])
    print("原始文档数量：", len(documents))
    print("筛选后文档数量：", len(res))

    state["documents"] = res

    return state


def drugs_filter(state):
    drugs = state["drugs"]
    question = state["question"]

    if drugs is None:
        return state

    res = []
    for drug in drugs:
        score = retrieval_grader.grade(question, drug)
        grade = score.binary_score
        if grade == "yes":
            res.append(drug)
    print("原始药品数量：", len(drugs))
    print("筛选后药品数量：", len(res))

    state["drugs"] = res
    return state


def disease_keyword(state):
    question = state["question"]
    res = disease_router.invoke(question)
    if res is None:
        return state
    state["keyword"] = res.disease
    return state


def search_drugs_by_name(state):
    keyword = state["keyword"]
    res = []
    data = pubmed_kg.invoke(keyword)
    drug_list = []
    for disease, info in data.items():
        drug_list.extend(info['药物治疗'])
    for drug in drug_list:
        drug_name = drug[0]
        drug_detail = searchByName(drug_name)
        if drug_detail is not None:
            res.append(drug_detail)
        # else:
        #     res.append(drug_name)

    # print("KG药品列表：" + str(res))
    state["drugs"] = res
    return state


def search_drugs_by_efficacy(state):
    keyword = state["keyword"]
    drugs = searchByEfficacy(keyword)
    state["drugs"].extend(drugs)
    return state


def generate_instruction(state):
    question = state["question"]
    documents = state["documents"]
    drugs = state["drugs"]

    instruction = med_instructor.invoke(question=question, documents=documents, drugs=drugs)
    state["instruction"] = instruction
    return state


workflow = StateGraph(State)

workflow.add_node("search_documents", search_documents)
workflow.add_node("documents_filter", documents_filter)
workflow.add_node("drugs_filter", drugs_filter)
workflow.add_node("disease_keyword", disease_keyword)
workflow.add_node("search_drugs_by_name", search_drugs_by_name)
workflow.add_node("search_drugs_by_efficacy", search_drugs_by_efficacy)
workflow.add_node("generate_instruction", generate_instruction)

workflow.add_edge(START, "search_documents")
workflow.add_edge("search_documents", "documents_filter")
workflow.add_edge("documents_filter", "disease_keyword")
workflow.add_edge("disease_keyword", "search_drugs_by_name")
workflow.add_edge("search_drugs_by_name", "search_drugs_by_efficacy")
workflow.add_edge("search_drugs_by_efficacy", "drugs_filter")
workflow.add_edge("drugs_filter", "generate_instruction")
workflow.add_edge("generate_instruction", END)

# Compile
app = workflow.compile()


if __name__ == '__main__':
    app.get_graph().draw_mermaid_png(output_file_path="../img/medication_instructor.png")

    inputs = {
        "question": "我最近有点咳嗽，吃点什么药好"
    }

    resp = app.invoke(inputs)
    print(resp)
    print(resp["instruction"].content)
