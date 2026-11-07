"""
Stage 6: Text Explanation Formatting & Input Assembly Library
Implements Section 3.6 and Section 4.3 of the paper:
Formats selected relations and UMLS concept definitions into natural language explanations (Figure 2),
and applies atomic non-cutting truncation within the 512-token budget (Section 4.3).
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
from biocuep.experiments import get_experiment_config, format_sample_for_experiment

logger = logging.getLogger(__name__)


class MetadataMethod(Enum):
    WITH_DEFINITION_RELATION = 1
    WITH_RELATION = 2
    WITH_DEFINITION = 3


def get_all_concepts_with_definition(context_ent, question_ent):
    all_concepts = {}
    for ent_list in (context_ent, question_ent):
        if not ent_list:
            continue
        for item in ent_list:
            if isinstance(item, dict):
                for k, v in item.items():
                    if isinstance(v, dict) and v.get("definition"):
                        all_concepts[k] = v["definition"]
    return all_concepts


def process_json(json_file_path, method: MetadataMethod, version_filter: str, kg: str = "both", max_tokens: int = 512):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    v_filter = "sm" if version_filter in ("similarity", "sm") else "qc"

    include_defs = method in (MetadataMethod.WITH_DEFINITION_RELATION, MetadataMethod.WITH_DEFINITION)
    include_rels = method in (MetadataMethod.WITH_DEFINITION_RELATION, MetadataMethod.WITH_RELATION)

    new_data = []
    for item in tqdm(data, desc=f"Formatting {version_filter} ({kg})"):
        question = item.get("question", "")
        context = item.get("context", "")

        defs = extract_definitions_from_sample(item) if include_defs else {}

        relations = []
        if include_rels:
            if kg in ("both", "primekg"):
                relations.extend(item.get(f"primekg_relevant_relations_{v_filter}", []))
            if kg in ("both", "hetionet"):
                relations.extend(item.get(f"hetionet_relevant_relations_{v_filter}", []))

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
        item["original_question"] = question
        new_data.append(item)

    return new_data


def process_dataset(dataset, method: MetadataMethod, version_filter: str, kg: str = "both", max_tokens: int = 512):
    v_dir = "similarity" if version_filter in ("similarity", "sm") else "qc"
    input_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / f"4_extracted_concepts_relations_filtered_{v_dir}"
    output_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / f"5_formatted_metadata_{method.name}_v{v_dir}_{kg}"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    for json_file_path in json_files:
        if json_file_path.name in ("train.json", "dev.json", "test.json"):
            new_data = process_json(str(json_file_path), method, version_filter, kg, max_tokens=max_tokens)
            out_file = output_dir / json_file_path.name
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)
            print(f"Saved formatted metadata to {out_file}")
