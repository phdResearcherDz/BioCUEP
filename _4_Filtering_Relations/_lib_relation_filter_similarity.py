"""
Stage 5: Similarity-based Relation Filtering Library
Implements Section 3.5, Equation (8) of the paper:
Rsim = Top-k(RS U RQ, Sim(vQ, vr))
Scores relation triplets using TAS-B dense retrieval encoder and selects top-k (k=10).
"""

import glob
import json
import os
from pathlib import Path
import logging

from config_file import PROJECT_ROOT
from biocuep.utils import tqdm
from biocuep.config import SIMILARITY_FILTER_TOP_K
from biocuep.filtering import filter_relations_similarity as core_filter_sim, load_tasb_model

logger = logging.getLogger(__name__)


def process_json(json_file_path, kg_name, top_k=SIMILARITY_FILTER_TOP_K, device="cpu"):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    tokenizer, model = load_tasb_model(device)

    new_data = []
    for item in tqdm(data, desc=f"Similarity Filtering {kg_name}"):
        filtered = core_filter_sim(
            item, kg_name, top_k=top_k, tokenizer=tokenizer, model=model, device=device
        )
        item[f"{kg_name}_relevant_relations_sm"] = filtered
        new_data.append(item)
    return new_data


def process_dataset(dataset, kg_name, top_k=SIMILARITY_FILTER_TOP_K, device="cpu"):
    input_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "3_extracted_concepts_relations"
    output_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "4_extracted_concepts_relations_filtered_similarity"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    for json_file_path in json_files:
        if json_file_path.name in ("train.json", "dev.json", "test.json"):
            new_data = process_json(str(json_file_path), kg_name, top_k=top_k, device=device)
            out_filename = f"{json_file_path.stem}_{kg_name}_{dataset}_with_relations_filtered.json"
            out_path = output_dir / out_filename
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)
            print(f"Saved similarity filtered data to {out_path}")
