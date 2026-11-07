"""
Dataset Preprocessing: BioASQ Formatter
Formats raw BioASQ JSON files (BioASQ 7B BLURB or BioASQ 12B 2024) into standard BioCUEP format.
Filters for Yes/No questions and joins gold evidence snippets as context S (Section 4.1).
"""

import json
import os
import argparse
from pathlib import Path
from config_file import PROJECT_ROOT


def extract_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    extracted_data = []
    questions_list = data.get("questions", data if isinstance(data, list) else [])

    for question in questions_list:
        if not isinstance(question, dict):
            continue
        q_type = question.get("type", "")
        # BioCUEP focuses on Yes/No questions (Section 4.1)
        if q_type == "yesno":
            question_text = question.get("body", question.get("question", ""))
            snippets = question.get("snippets", [])
            if isinstance(snippets, list):
                context = " ".join(s.get("text", "") for s in snippets if isinstance(s, dict))
            else:
                context = str(question.get("context", ""))

            answer = question.get("exact_answer", question.get("answer", None))
            q_id = question.get("id", "")

            extracted_data.append({
                "id": q_id,
                "question": question_text,
                "context": context,
                "answer": answer,
                "type": q_type,
            })

    return extracted_data


def process_dataset(input_path: str, output_path: str):
    p = Path(input_path)
    all_data = []
    if p.is_dir():
        for f in p.glob("*.json"):
            all_data.extend(extract_data(str(f)))
    elif p.is_file():
        all_data = extract_data(str(p))
    else:
        print(f"Path does not exist: {input_path}")
        return

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as outfile:
        json.dump(all_data, outfile, indent=4)
    print(f"Saved {len(all_data)} Yes/No questions to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess BioASQ datasets for BioCUEP")
    parser.add_argument("--input", type=str, default=str(PROJECT_ROOT / "data" / "raw_bioasq.json"))
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "Pre_Processed_Datasets" / "BioASQ2024" / "0_Preprocessed" / "train.json"))
    args = parser.parse_args()

    if Path(args.input).exists():
        process_dataset(args.input, args.output)
    else:
        print(f"Input file not found at {args.input}. Ready for user data.")
