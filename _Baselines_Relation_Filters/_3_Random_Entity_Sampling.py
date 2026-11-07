"""
Baseline: Random Entity Sampling
Randomly samples entities and retrieves their associated relations.
"""

import glob
import json
import os
import random
from pathlib import Path
from config_file import PROJECT_ROOT
from biocuep.utils import tqdm

datasets = ["BioASQ_BLURB", "BioASQ2024", "PubmedQA"]


def get_random_entities_relations(entity_relations, num_entities=10):
    if len(entity_relations) <= num_entities:
        selected_relations = []
        for ent in entity_relations.values():
            if isinstance(ent, dict):
                selected_relations.extend(ent.get("triples", []))
        return selected_relations

    sampled_keys = random.sample(list(entity_relations.keys()), num_entities)
    selected_relations = []
    for k in sampled_keys:
        ent = entity_relations[k]
        if isinstance(ent, dict):
            selected_relations.extend(ent.get("triples", []))

    return selected_relations


def process_json_random_entity_sampling(json_file_path, kg_name, num_entities=10):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    new_data = []
    for item in tqdm(data, desc=f"Random Entity Sampling {kg_name}"):
        q_ents = item.get(f"{kg_name}_concepts_question_with_kg_data", {}) or {}
        ctx_ents = item.get(f"{kg_name}_concepts_context_with_kg_data", {}) or {}

        q_rels = get_random_entities_relations(q_ents, num_entities)
        ctx_rels = get_random_entities_relations(ctx_ents, num_entities)

        item[f"{kg_name}_random_entity_relations"] = q_rels + ctx_rels
        new_data.append(item)

    return new_data


def process_dataset_random_entity_sampling(dataset, kg_name, num_entities=10):
    input_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "3_extracted_concepts_relations"
    output_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "4_extracted_concepts_relations_random_entity"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    for json_file_path in json_files:
        if json_file_path.name in ("train.json", "dev.json", "test.json"):
            new_data = process_json_random_entity_sampling(str(json_file_path), kg_name, num_entities)
            out_filename = f"{json_file_path.stem}_{kg_name}_{dataset}_with_random_entity_relations.json"
            out_path = output_dir / out_filename
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)
            print(f"Saved random entity sampling relations to {out_path}")


def filter_relations_random_entity(target_datasets=None):
    ds_list = target_datasets or datasets
    kgs = ["primekg", "hetionet"]
    for dataset in ds_list:
        for kg_name in kgs:
            process_dataset_random_entity_sampling(dataset, kg_name, num_entities=10)


if __name__ == "__main__":
    filter_relations_random_entity()
