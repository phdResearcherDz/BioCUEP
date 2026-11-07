"""
Baseline 4: TF-IDF Weighted Relations (TF-IDF-RI)
Implements Section 6.1 and Table 6 of the paper:
Ranks relation triplets by TF-IDF lexical similarity to the question (N=10).
"""

import glob
import json
import os
from pathlib import Path
from config_file import PROJECT_ROOT
from biocuep.utils import tqdm
from biocuep.baselines import filter_tfidf_ri

datasets = ["BioASQ_BLURB", "BioASQ2024", "PubmedQA"]


def process_json_tfidf(json_file_path, kg_name, top_n=10):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    new_data = []
    for item in tqdm(data, desc=f"TF-IDF-RI {kg_name}"):
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

        # Correctly format relation texts and compute TF-IDF against question
        top_relations = filter_tfidf_ri(all_relations, query=query, n=top_n)
        item[f"{kg_name}_tfidf_relations"] = top_relations
        new_data.append(item)

    return new_data


def process_dataset_tfidf(dataset, kg_name, top_n=10):
    input_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "3_extracted_concepts_relations"
    output_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "4_extracted_concepts_relations_tfidf"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    for json_file_path in json_files:
        if json_file_path.name in ("train.json", "dev.json", "test.json"):
            new_data = process_json_tfidf(str(json_file_path), kg_name, top_n)
            out_filename = f"{json_file_path.stem}_{kg_name}_{dataset}_with_tfidf_relations.json"
            out_path = output_dir / out_filename
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)
            print(f"Saved TF-IDF relations to {out_path}")


def filter_relations_tfidf(target_datasets=None):
    ds_list = target_datasets or datasets
    kgs = ["primekg", "hetionet"]
    for dataset in ds_list:
        for kg_name in kgs:
            process_dataset_tfidf(dataset, kg_name, top_n=10)


if __name__ == "__main__":
    filter_relations_tfidf()
