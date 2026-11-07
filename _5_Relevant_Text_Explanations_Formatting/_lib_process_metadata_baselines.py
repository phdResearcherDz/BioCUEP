"""
Baseline Formatting Engine
Formats baseline relations (random, top_n, clustered, tfidf, dense_retrieval) into reader input format.
"""

from enum import Enum
import glob
import json
import os
from pathlib import Path
import logging

from config_file import PROJECT_ROOT
from biocuep.utils import tqdm
from biocuep.formatting import (
    format_relations_to_text,
    format_definitions_to_text,
    extract_definitions_from_sample,
    assemble_enriched_input
)

logger = logging.getLogger(__name__)


class MetadataMethod(Enum):
    WITH_DEFINITION_RELATION = 1
    WITH_RELATION = 2
    WITH_DEFINITION = 3


def process_json(json_file_path, method: MetadataMethod, baseline_type: str, kg: str = "both", max_tokens: int = 512):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    include_defs = method in (MetadataMethod.WITH_DEFINITION_RELATION, MetadataMethod.WITH_DEFINITION)
    include_rels = method in (MetadataMethod.WITH_DEFINITION_RELATION, MetadataMethod.WITH_RELATION)

    # Key suffix mapping for baseline methods
    baseline_key_map = {
        "random": "random_relations",
        "top_n": "top_n_relations",
        "clustered": "cluster_based_relations",
        "tfidf": "tfidf_relations",
        "dense_retrieval": "dense_relations"
    }
    rel_key = baseline_key_map.get(baseline_type, f"{baseline_type}_relations")

    new_data = []
    for item in tqdm(data, desc=f"Formatting Baseline {baseline_type}"):
        question = item.get("question", "")
        context = item.get("context", "")
        defs = extract_definitions_from_sample(item) if include_defs else {}

        relations = []
        if include_rels:
            for kg_name in (["primekg", "hetionet"] if kg == "both" else [kg]):
                relations.extend(item.get(f"{kg_name}_{rel_key}", []))

        assembly = assemble_enriched_input(
            question=question,
            context=context,
            relations=relations,
            definitions=defs,
            max_tokens=max_tokens,
            include_relations=include_rels,
            include_definitions=include_defs
        )

        item["metadata"] = assembly["metadata"]
        item["enriched_input"] = assembly["enriched_input"]
        new_data.append(item)

    return new_data


def process_dataset(dataset, method: MetadataMethod, baseline_type: str, kg: str = "both", max_tokens: int = 512):
    dir_name_map = {
        "random": "4_extracted_concepts_relations_random",
        "top_n": "4_extracted_concepts_relations_top_n",
        "clustered": "4_extracted_concepts_relations_clustered",
        "tfidf": "4_extracted_concepts_relations_tfidf",
        "dense_retrieval": "4_extracted_concepts_relations_dense_retrieval"
    }
    in_dir_name = dir_name_map.get(baseline_type, f"4_extracted_concepts_relations_{baseline_type}")
    input_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / in_dir_name
    output_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / f"5_formatted_baseline_{baseline_type}_{method.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    for json_file_path in json_files:
        if json_file_path.name in ("train.json", "dev.json", "test.json"):
            new_data = process_json(str(json_file_path), method, baseline_type, kg, max_tokens=max_tokens)
            out_file = output_dir / json_file_path.name
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)
            print(f"Saved formatted baseline to {out_file}")
