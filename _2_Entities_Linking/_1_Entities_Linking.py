"""
Stage 3 Runner: Biomedical Entity Linking to Hetionet and PrimeKG
"""

import json
import os
from pathlib import Path
from config_file import PROJECT_ROOT
from _2_Entities_Linking._lib_entities_linking import (
    connect_to_neo4j,
    fetch_nodes,
    process_dataset,
    HETIONET_NEO4J_URI,
    HETIONET_NEO4J_USER,
    HETIONET_NEO4J_PASSWORD,
    PRIMEKG_NEO4J_URI,
    PRIMEKG_NEO4J_USER,
    PRIMEKG_NEO4J_PASSWORD
)

datasets = ["BioASQ_BLURB", "BioASQ2024", "PubmedQA"]


def ProcessPrimeKG(target_datasets=None):
    ds_list = target_datasets or datasets
    driver = connect_to_neo4j(PRIMEKG_NEO4J_URI, PRIMEKG_NEO4J_USER, PRIMEKG_NEO4J_PASSWORD)
    df_nodes = fetch_nodes(driver, "node_name")
    for dataset in ds_list:
        process_dataset(dataset, df_nodes, "primekg", "node_name")


def ProcessHetionet(target_datasets=None):
    ds_list = target_datasets or datasets
    driver = connect_to_neo4j(HETIONET_NEO4J_URI, HETIONET_NEO4J_USER, HETIONET_NEO4J_PASSWORD)
    df_nodes = fetch_nodes(driver, "name")
    for dataset in ds_list:
        process_dataset(dataset, df_nodes, "hetionet", "name")


def merge_records(data1, data2, kg1="primekg", kg2="hetionet"):
    merged_data = []
    for ind in range(min(len(data1), len(data2))):
        item = data1[ind]
        item2 = data2[ind]
        item[f"{kg2}_concepts_context"] = item2.get(f"{kg2}_concepts_context", {})
        item[f"{kg2}_concepts_question"] = item2.get(f"{kg2}_concepts_question", {})
        merged_data.append(item)
    return merged_data


def entities_linking(target_datasets=None):
    ds_list = target_datasets or datasets
    print("=== Processing Hetionet Linking ===")
    ProcessHetionet(ds_list)

    print("\n=== Processing PrimeKG Linking ===")
    ProcessPrimeKG(ds_list)

    print("\n=== Merging Linked Concepts ===")
    kgs = ["primekg", "hetionet"]
    for dataset in ds_list:
        dir_path = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "2_kg_linked_entities"
        if not dir_path.exists():
            continue

        for split in ["train", "dev", "test"]:
            file1 = dir_path / f"{split}_{kgs[0]}_{dataset}_with_concept.json"
            file2 = dir_path / f"{split}_{kgs[1]}_{dataset}_with_concept.json"

            if file1.exists() and file2.exists():
                with open(file1, 'r', encoding='utf-8') as f:
                    d1 = json.load(f)
                with open(file2, 'r', encoding='utf-8') as f:
                    d2 = json.load(f)
                merged = merge_records(d1, d2, kg1=kgs[0], kg2=kgs[1])
                out_path = dir_path / f"{split}.json"
                with open(out_path, 'w', encoding='utf-8') as outfile:
                    json.dump(merged, outfile, indent=4)
                print(f"Saved merged {split} to {out_path}")


if __name__ == "__main__":
    entities_linking()
