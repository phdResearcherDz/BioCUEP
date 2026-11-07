"""
Stage 4: Semantic Relation Extraction Library
Implements Section 3.4 of the paper:
Extracts first-level semantic relations from Hetionet and PrimeKG for concepts in LS and LQ.
"""

import glob
import json
import os
from pathlib import Path
import logging

from config_file import PROJECT_ROOT
from biocuep.utils import tqdm
from biocuep.config import (
    HETIONET_NEO4J_URI,
    HETIONET_NEO4J_USER,
    HETIONET_NEO4J_PASSWORD,
    PRIMEKG_NEO4J_URI,
    PRIMEKG_NEO4J_USER,
    PRIMEKG_NEO4J_PASSWORD
)
from biocuep.relations import (
    connect_to_neo4j,
    fetch_first_level_triples_neo4j,
    extract_relations_for_concepts,
    extract_relations_for_sample
)

logger = logging.getLogger(__name__)


def process_json(driver, json_file_path, kg_name, node_name_attribute="name"):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    new_data = []
    for item in tqdm(data, desc=f"Extracting {kg_name} relations"):
        updated_item = extract_relations_for_sample(
            item, kg_name=kg_name, driver=driver, node_name_attribute=node_name_attribute
        )
        new_data.append(updated_item)

    return new_data


def process_dataset(driver, dataset, kg_name, node_name_attribute="name"):
    input_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "2_kg_linked_entities"
    output_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "3_extracted_concepts_relations"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    for json_file_path in json_files:
        if json_file_path.name in ("train.json", "dev.json", "test.json"):
            new_data = process_json(driver, str(json_file_path), kg_name, node_name_attribute)
            out_filename = f"{json_file_path.stem}_{kg_name}_{dataset}_with_relations.json"
            out_path = output_dir / out_filename
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)
            print(f"Saved relations to {out_path}")
