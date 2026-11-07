"""
Dataset Preprocessing: PubMedQA Formatter
Formats raw PubMedQA files into standard BioCUEP format.
Extracts abstract context (without conclusion) and answers (yes, no, maybe) (Section 4.1).
"""

import json
import os
import argparse
from pathlib import Path
from config_file import PROJECT_ROOT


def extract_data_from_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    extracted_data = []
    for key, value in data.items():
        question_text = value.get("QUESTION", "")
        # Context without conclusion as gold evidence segment (Section 4.1)
        contexts = value.get("CONTEXTS", [])
        context = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
        answer = value.get("final_decision", value.get("decision", ""))

        extracted_data.append({
            "id": key,
            "question": question_text,
            "context": context,
            "answer": answer,
            "type": "yesno_maybe"
        })

    return extracted_data


def process_dataset(input_file: str, output_file: str):
    p = Path(input_file)
    if not p.exists():
        print(f"File not found: {input_file}")
        return

    extracted = extract_data_from_file(str(p))
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as outfile:
        json.dump(extracted, outfile, indent=4)
    print(f"Saved {len(extracted)} samples to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess PubMedQA dataset for BioCUEP")
    parser.add_argument("--input", type=str, default=str(PROJECT_ROOT / "data" / "ori_pqal.json"))
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "Pre_Processed_Datasets" / "PubmedQA" / "0_Preprocessed" / "data.json"))
    args = parser.parse_args()

    if Path(args.input).exists():
        process_dataset(args.input, args.output)
    else:
        print(f"Input file not found at {args.input}. Ready for user data.")
