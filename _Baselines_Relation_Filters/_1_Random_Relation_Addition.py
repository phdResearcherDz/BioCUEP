"""
Baseline 1: RRI (Random Relation Integration)
Implements Section 6.1 and Table 6 of the paper:
Randomly integrates biomedical relations from the knowledge graph (k=10) to assess the effect of unfiltered relation inclusion.
"""

import glob
import json
import os
import random
from pathlib import Path
from config_file import PROJECT_ROOT
from biocuep.utils import tqdm
from biocuep.baselines import filter_rri_random

datasets = ["BioASQ_BLURB", "BioASQ2024", "PubmedQA"]


def process_json_random(json_file_path, kg_name, num_relations=10, seed=42):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    new_data = []
    for item in tqdm(data, desc=f"RRI Random {kg_name}"):
        question_relations = item.get(f"{kg_name}_concepts_question_with_kg_data", {}) or {}
        context_relations = item.get(f"{kg_name}_concepts_context_with_kg_data", {}) or {}

        all_relations = []
        for cr in question_relations.values():
            if isinstance(cr, dict):
                all_relations.extend(cr.get("triples", []))
        for cr in context_relations.values():
            if isinstance(cr, dict):
                all_relations.extend(cr.get("triples", []))

        random_rels = filter_rri_random(all_relations, num_relations=num_relations, seed=seed)
        item[f"{kg_name}_random_relations"] = random_rels
        new_data.append(item)

    return new_data


def process_dataset_random(dataset, kg_name, num_relations=10, seed=42):
    input_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "3_extracted_concepts_relations"
    output_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "4_extracted_concepts_relations_random"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    for json_file_path in json_files:
        if json_file_path.name in ("train.json", "dev.json", "test.json"):
            new_data = process_json_random(str(json_file_path), kg_name, num_relations, seed=seed)
            out_filename = f"{json_file_path.stem}_{kg_name}_{dataset}_with_random_relations.json"
            out_path = output_dir / out_filename
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)
            print(f"Saved random relations to {out_path}")


def filter_relations_random(target_datasets=None):
    ds_list = target_datasets or datasets
    kgs = ["primekg", "hetionet"]
    for dataset in ds_list:
        for kg_name in kgs:
            process_dataset_random(dataset, kg_name, num_relations=10)


if __name__ == "__main__":
    filter_relations_random()
