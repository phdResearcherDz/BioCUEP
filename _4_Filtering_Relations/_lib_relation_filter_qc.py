"""
Stage 5: QC Relation Filtering Library (Entity-Link Filtering)
Implements Section 3.5, Equation (7) of the paper:
RSQ = {(h, r, t) in RS U RQ | h, t in LS U LQ}
Retains only relations that directly connect concepts found in both context and question.
"""

import glob
import json
import os
from pathlib import Path
import logging

from config_file import PROJECT_ROOT
from biocuep.utils import tqdm
from biocuep.filtering import filter_relations_qc as core_filter_qc

logger = logging.getLogger(__name__)


def process_json(json_file_path, kg_name):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    new_data = []
    for item in tqdm(data, desc=f"QC Filtering {kg_name}"):
        filtered = core_filter_qc(item, kg_name)
        item[f"{kg_name}_relevant_relations_qc"] = filtered
        new_data.append(item)
    return new_data


def process_dataset(dataset, kg_name):
    input_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "3_extracted_concepts_relations"
    output_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "4_extracted_concepts_relations_filtered_qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    for json_file_path in json_files:
        if json_file_path.name in ("train.json", "dev.json", "test.json"):
            new_data = process_json(str(json_file_path), kg_name)
            out_filename = f"{json_file_path.stem}_{kg_name}_{dataset}_with_relations_filtered.json"
            out_path = output_dir / out_filename
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)
            print(f"Saved QC filtered data to {out_path}")
