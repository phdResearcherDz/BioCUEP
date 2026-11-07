"""
Baseline 3: Cluster-based Relation Selection
Implements Section 6.1 and Table 6 of the paper:
Clusters relation triplets using K-Means (k=10) on relation embeddings and selects the relation closest to each centroid.
"""

import glob
import json
import os
from pathlib import Path
from config_file import PROJECT_ROOT
from biocuep.utils import tqdm
from biocuep.baselines import filter_cluster_based

datasets = ["BioASQ_BLURB", "BioASQ2024", "PubmedQA"]


def process_json_cluster_based(json_file_path, kg_name, num_clusters=10, seed=42):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    new_data = []
    for item in tqdm(data, desc=f"Cluster-based {kg_name}"):
        question_relations = item.get(f"{kg_name}_concepts_question_with_kg_data", {}) or {}
        context_relations = item.get(f"{kg_name}_concepts_context_with_kg_data", {}) or {}

        all_relations = []
        for cr in question_relations.values():
            if isinstance(cr, dict):
                all_relations.extend(cr.get("triples", []))
        for cr in context_relations.values():
            if isinstance(cr, dict):
                all_relations.extend(cr.get("triples", []))

        # Clusters relation embeddings and selects closest to each centroid (k=10)
        selected_relations = filter_cluster_based(all_relations, k=num_clusters, seed=seed)
        item[f"{kg_name}_cluster_based_relations"] = selected_relations
        new_data.append(item)

    return new_data


def process_dataset_cluster_based(dataset, kg_name, num_clusters=10, seed=42):
    input_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "3_extracted_concepts_relations"
    output_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "4_extracted_concepts_relations_clustered"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    for json_file_path in json_files:
        if json_file_path.name in ("train.json", "dev.json", "test.json"):
            new_data = process_json_cluster_based(str(json_file_path), kg_name, num_clusters, seed=seed)
            out_filename = f"{json_file_path.stem}_{kg_name}_{dataset}_with_clustered_relations.json"
            out_path = output_dir / out_filename
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)
            print(f"Saved clustered relations to {out_path}")


def filter_relations_clustered(target_datasets=None):
    ds_list = target_datasets or datasets
    kgs = ["primekg", "hetionet"]
    for dataset in ds_list:
        for kg_name in kgs:
            process_dataset_cluster_based(dataset, kg_name, num_clusters=10)


if __name__ == "__main__":
    filter_relations_clustered()
