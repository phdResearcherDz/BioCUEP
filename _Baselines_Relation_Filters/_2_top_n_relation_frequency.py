"""
Baseline 2: Top-N Relation Frequency
Implements Section 6.1 and Table 6 of the paper:
Keeps the most N frequent relations (N=10).
"""

import glob
import json
import os
from pathlib import Path
from config_file import PROJECT_ROOT
from biocuep.utils import tqdm
from biocuep.baselines import filter_top_n_frequency

datasets = ["BioASQ_BLURB", "BioASQ2024", "PubmedQA"]


def process_json_top_n(json_file_path, kg_name, num_relations=10):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    new_data = []
    for item in tqdm(data, desc=f"Top-N {kg_name}"):
        question_relations = item.get(f"{kg_name}_concepts_question_with_kg_data", {}) or {}
        context_relations = item.get(f"{kg_name}_concepts_context_with_kg_data", {}) or {}

        all_relations = []
        for cr in question_relations.values():
            if isinstance(cr, dict):
                all_relations.extend(cr.get("triples", []))
        for cr in context_relations.values():
            if isinstance(cr, dict):
                all_relations.extend(cr.get("triples", []))

        top_relations = filter_top_n_frequency(all_relations, n=num_relations)
        item[f"{kg_name}_top_n_relations"] = top_relations
        new_data.append(item)

    return new_data


def process_dataset_top_n(dataset, kg_name, num_relations=10):
    input_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "3_extracted_concepts_relations"
    output_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "4_extracted_concepts_relations_top_n"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    for json_file_path in json_files:
        if json_file_path.name in ("train.json", "dev.json", "test.json"):
            new_data = process_json_top_n(str(json_file_path), kg_name, num_relations)
            out_filename = f"{json_file_path.stem}_{kg_name}_{dataset}_with_top_n_relations.json"
            out_path = output_dir / out_filename
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)
            print(f"Saved Top-N relations to {out_path}")


def filter_relations_top_n(target_datasets=None):
    ds_list = target_datasets or datasets
    kgs = ["primekg", "hetionet"]
    for dataset in ds_list:
        for kg_name in kgs:
            process_dataset_top_n(dataset, kg_name, num_relations=10)


if __name__ == "__main__":
    filter_relations_top_n()
