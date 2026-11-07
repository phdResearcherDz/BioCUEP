"""
Stage 1 & 2: Biomedical Entity Extraction and UMLS Normalization
Implements Section 3.1 & 3.2 of the paper:
1. Multi-model SciSpaCy NER extraction (CRAFT, JNLPBA, BC5CDR, BIONLP13CG) with post-extraction filter F.
2. Approximate candidate retrieval in UMLS using char-3grams: Ce = ApproxMatch(e, UMLS).
3. Candidate scoring via cosine similarity against BioBERT embedding of 10-word sliding window:
   c*_e = argmax_{c in Ce} Sim(e, c) (Equation 3).
4. UMLS concept definition retrieval.
"""

import json
import os
import re
import sys
from pathlib import Path
from config_file import PROJECT_ROOT
from biocuep.utils import tqdm, load_dataset_file
from biocuep.extraction import (
    extract_entities_from_text,
    filtering_function,
    is_full_word
)
from biocuep.normalization import (
    normalize_entities,
    get_local_context_window,
    disambiguate_concept,
    load_biobert
)

list_datasets = ["BioASQ_BLURB", "BioASQ2024", "PubmedQA"]


def process_json_file(file_path, output_directory, window_size=10, device="cpu"):
    try:
        data = load_dataset_file(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    new_data = []
    print(f"Processing {len(data)} items from {file_path}...")

    # Load BioBERT if available
    tokenizer, model = load_biobert(device)

    for item in tqdm(data, desc="Extracting & Normalizing"):
        context = item.get('context', '')
        question = item.get('question', '')

        # Stage 1: Extraction
        raw_ctx_ents = extract_entities_from_text(context)
        raw_q_ents = extract_entities_from_text(question)

        # Stage 2: Normalization with 10-word sliding window BioBERT argmax
        norm_ctx_ents = normalize_entities(raw_ctx_ents, context, tokenizer=tokenizer, model=model, window_size=window_size, device=device)
        norm_q_ents = normalize_entities(raw_q_ents, question, tokenizer=tokenizer, model=model, window_size=window_size, device=device)

        item['context_entities'] = norm_ctx_ents
        item['question_entities'] = norm_q_ents
        new_data.append(item)

    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, os.path.basename(file_path))

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, indent=4)
    print(f"Saved normalized data to {output_file_path}")


def entities_normalization(datasets=None):
    target_datasets = datasets or list_datasets
    for dataset in target_datasets:
        path_dataset = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset
        input_directory = path_dataset / "0_Preprocessed"
        output_directory = path_dataset / "1_extracted_entities_with_metadata"

        # Check Data/ folder from Data.zip if Pre_Processed_Datasets does not yet exist
        if not input_directory.exists():
            alt_names = [dataset, dataset.replace("_BLURB", "_Blurb"), dataset.replace("_Blurb", "_BLURB")]
            for alt in alt_names:
                alt_dir = PROJECT_ROOT / "Data" / alt
                if alt_dir.exists():
                    input_directory = alt_dir
                    break

        if not input_directory.exists():
            print(f"Skipping {dataset}: No data found in Pre_Processed_Datasets/ or Data/")
            continue

        print(f"\n=== Processing Dataset: {dataset} (Source: {input_directory}) ===")
        for filename in os.listdir(input_directory):
            if filename.endswith(".json"):
                file_path = input_directory / filename
                process_json_file(str(file_path), str(output_directory))


if __name__ == '__main__':
    entities_normalization()
