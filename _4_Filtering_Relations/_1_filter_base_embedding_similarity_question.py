"""
Stage 5 Runner: Similarity-based Relation Filtering (TAS-B Top-k, k=10)
"""

import json
import os
from pathlib import Path
from config_file import PROJECT_ROOT
from _4_Filtering_Relations._lib_relation_filter_similarity import process_dataset

datasets = ["BioASQ_BLURB", "BioASQ2024", "PubmedQA"]


def ProcessPrimeKG(target_datasets=None):
    ds_list = target_datasets or datasets
    for dataset in ds_list:
        process_dataset(dataset, kg_name="primekg")


def ProcessHetionet(target_datasets=None):
    ds_list = target_datasets or datasets
    for dataset in ds_list:
        process_dataset(dataset, kg_name="hetionet")


def merge_records(data1, data2, kg1="primekg", kg2="hetionet"):
    merged_data = []
    for ind in range(min(len(data1), len(data2))):
        item = data1[ind]
        item2 = data2[ind]
        item[f"{kg2}_relevant_relations_sm"] = item2.get(f"{kg2}_relevant_relations_sm", [])
        merged_data.append(item)
    return merged_data


def filter_relations_sm(target_datasets=None):
    ds_list = target_datasets or datasets
    print("=== Running Similarity-based Filter on PrimeKG ===")
    ProcessPrimeKG(ds_list)

    print("\n=== Running Similarity-based Filter on Hetionet ===")
    ProcessHetionet(ds_list)

    print("\n=== Merging Filtered Similarity Data ===")
    kgs = ["primekg", "hetionet"]
    for dataset in ds_list:
        dir_path = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "4_extracted_concepts_relations_filtered_similarity"
        if not dir_path.exists():
            continue

        for split in ["train", "dev", "test"]:
            file1 = dir_path / f"{split}_{kgs[0]}_{dataset}_with_relations_filtered.json"
            file2 = dir_path / f"{split}_{kgs[1]}_{dataset}_with_relations_filtered.json"

            if file1.exists() and file2.exists():
                with open(file1, 'r', encoding='utf-8') as f:
                    d1 = json.load(f)
                with open(file2, 'r', encoding='utf-8') as f:
                    d2 = json.load(f)

                merged = merge_records(d1, d2, kg1=kgs[0], kg2=kgs[1])
                out_path = dir_path / f"{split}.json"
                with open(out_path, 'w', encoding='utf-8') as outfile:
                    json.dump(merged, outfile, indent=4)
                print(f"Saved merged similarity {split} to {out_path}")


if __name__ == "__main__":
    filter_relations_sm()
