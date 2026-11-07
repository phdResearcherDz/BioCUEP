"""
Baseline: Dense Retrieval for Relation Ranking
Ranks relation triplets by dense embedding similarity to the question using SentenceTransformer / TAS-B (top-10).
"""

import glob
import json
import os
from pathlib import Path
from config_file import PROJECT_ROOT
from biocuep.utils import tqdm
from biocuep.baselines import relation_to_text

datasets = ["BioASQ_BLURB", "BioASQ2024", "PubmedQA"]


def dense_retrieval_relation_ranking(all_relations, query, model=None, top_n=10):
    if not all_relations or not query:
        return []

    if len(all_relations) <= top_n:
        return all_relations

    relation_texts = [relation_to_text(r) for r in all_relations]

    try:
        from sentence_transformers import SentenceTransformer, util
        if model is None:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        query_embedding = model.encode(query, convert_to_tensor=True)
        relation_embeddings = model.encode(relation_texts, convert_to_tensor=True)

        scores = util.pytorch_cos_sim(query_embedding, relation_embeddings).squeeze()
        top_indices = scores.argsort(descending=True)[:top_n]
        return [all_relations[i] for i in top_indices.tolist()]
    except Exception as ex:
        return all_relations[:top_n]


def process_json_dense_retrieval(json_file_path, kg_name, top_n=10):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    new_data = []
    for item in tqdm(data, desc=f"Dense Retrieval {kg_name}"):
        query = item.get("question", item.get("query", ""))
        question_relations = item.get(f"{kg_name}_concepts_question_with_kg_data", {}) or {}
        context_relations = item.get(f"{kg_name}_concepts_context_with_kg_data", {}) or {}

        all_relations = []
        for cr in question_relations.values():
            if isinstance(cr, dict):
                all_relations.extend(cr.get("triples", []))
        for cr in context_relations.values():
            if isinstance(cr, dict):
                all_relations.extend(cr.get("triples", []))

        top_relations = dense_retrieval_relation_ranking(all_relations, query, top_n=top_n)
        item[f"{kg_name}_dense_relations"] = top_relations
        new_data.append(item)

    return new_data


def process_dataset_dense_retrieval(dataset, kg_name, top_n=10):
    input_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "3_extracted_concepts_relations"
    output_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "4_extracted_concepts_relations_dense_retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    for json_file_path in json_files:
        if json_file_path.name in ("train.json", "dev.json", "test.json"):
            new_data = process_json_dense_retrieval(str(json_file_path), kg_name, top_n)
            out_filename = f"{json_file_path.stem}_{kg_name}_{dataset}_with_dense_relations.json"
            out_path = output_dir / out_filename
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)
            print(f"Saved dense retrieval relations to {out_path}")


def filter_relations_dense(target_datasets=None):
    ds_list = target_datasets or datasets
    kgs = ["primekg", "hetionet"]
    for dataset in ds_list:
        for kg_name in kgs:
            process_dataset_dense_retrieval(dataset, kg_name, top_n=10)


if __name__ == "__main__":
    filter_relations_dense()
