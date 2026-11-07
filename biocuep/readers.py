"""
Downstream Biomedical Readers & Evaluation Runners
Implements Section 4.3 and Section 5 of the paper:
1. Fine-tuned classification reader (BioLinkBERT, GPT-2, T5):
   - |Y| = 2 for BioASQ ({yes, no})
   - |Y| = 3 for PubMedQA ({yes, no, maybe})
2. Few-shot prompt constructor and normalized answer extraction:
   - 3 fixed demonstrations with controlled output pattern
   - Lowercasing, punctuation stripping, first definitive token extraction
"""

import re
import string
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# Official label spaces
LABEL_SPACES = {
    "BioASQ_BLURB": {"yes", "no"},
    "BioASQ2024": {"yes", "no"},
    "PubmedQA": {"yes", "no", "maybe"},
}

LABEL_TO_ID = {
    "yes": 1,
    "no": 0,
    "maybe": 2
}

ID_TO_LABEL = {
    1: "yes",
    0: "no",
    2: "maybe"
}

# 3-Shot Fixed Demonstrations (Section 4.3)
FEW_SHOT_DEMONSTRATIONS = {
    "BioASQ": [
        {
            "question": "Is Hirschsprung disease a congenital anomaly?",
            "context": "Hirschsprung disease is a birth defect in which nerve cells are missing at the end of a child's bowel.",
            "answer": "yes"
        },
        {
            "question": "Is actin an extracellular matrix protein?",
            "context": "Actin is an abundant intracellular protein that forms microfilaments in eukaryotic cells.",
            "answer": "no"
        },
        {
            "question": "Does Aspirin inhibit cyclooxygenase?",
            "context": "Aspirin acts as an irreversible inhibitor of cyclooxygenase enzymes COX-1 and COX-2.",
            "answer": "yes"
        }
    ],
    "PubmedQA": [
        {
            "question": "Does physical exercise reduce cardiovascular mortality?",
            "context": "Long-term cohort studies demonstrate significant reduction in cardiovascular deaths among active individuals.",
            "answer": "yes"
        },
        {
            "question": "Is antibiotic treatment effective against acute viral bronchitis?",
            "context": "Clinical trials show no significant benefit of antibiotic therapy over placebo in acute uncomplicated viral bronchitis.",
            "answer": "no"
        },
        {
            "question": "Can vitamin D supplementation prevent autoimmune thyroiditis?",
            "context": "Several observational studies suggest correlation, but randomized prospective trials remain inconclusive and contradictory.",
            "answer": "maybe"
        }
    ]
}


def normalize_answer_token(raw_output: str, admissible_labels: Set[str]) -> Optional[str]:
    """
    Extracts and normalizes the first definitive answer token per Section 4.3:
    1. Lowercases the string.
    2. Strips punctuation.
    3. Finds first occurrence of any admissible label.
    """
    if not raw_output:
        return None

    clean = raw_output.lower()
    # Strip leading markdown, punctuation, and labels like 'answer:'
    clean = re.sub(r'^(?:answer|the answer is|response|prediction)\s*[:\-]\s*', '', clean, flags=re.IGNORECASE)
    clean = clean.translate(str.maketrans('', '', string.punctuation)).strip()

    tokens = clean.split()
    for tok in tokens:
        if tok in admissible_labels:
            return tok

    return None


def construct_few_shot_prompt(
    question: str,
    context: str,
    metadata_text: str = "",
    dataset_name: str = "BioASQ_BLURB",
    demos: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Builds 3-shot prompt using prompt pattern control (Section 4.3 & [48]).
    Includes question, context, enriched metadata, and strictly defines admissible labels.
    """
    is_pubmedqa = "pubmed" in dataset_name.lower()
    admissible_str = "yes, no, or maybe" if is_pubmedqa else "yes or no"
    demo_key = "PubmedQA" if is_pubmedqa else "BioASQ"

    if demos is None:
        demos = FEW_SHOT_DEMONSTRATIONS.get(demo_key, FEW_SHOT_DEMONSTRATIONS["BioASQ"])

    prompt_lines = [
        f"You are an expert biomedical QA assistant. Answer the following medical question with exactly one word from [{admissible_str}] based on the provided context and biomedical knowledge.",
        ""
    ]

    # Add 3 demonstrations
    for i, demo in enumerate(demos, 1):
        prompt_lines.append(f"Example {i}:")
        prompt_lines.append(f"Context: {demo['context']}")
        prompt_lines.append(f"Question: {demo['question']}")
        prompt_lines.append(f"Answer: {demo['answer']}")
        prompt_lines.append("")

    prompt_lines.append("Now answer the target question:")
    if metadata_text:
        prompt_lines.append(f"Enriched Medical Knowledge: {metadata_text}")
    prompt_lines.append(f"Context: {context}")
    prompt_lines.append(f"Question: {question}")
    prompt_lines.append("Answer:")

    return "\n".join(prompt_lines)


def evaluate_predictions(
    ground_truth: List[str],
    predictions: List[Optional[str]]
) -> Dict[str, float]:
    """
    Computes strict accuracy: exact match between predicted and gold label.
    Unmapped or incorrect answers are counted as wrong (Section 4.3).
    """
    total = len(ground_truth)
    if total == 0:
        return {"accuracy": 0.0, "total": 0, "correct": 0}

    correct = 0
    for gt, pred in zip(ground_truth, predictions):
        if pred is not None and str(gt).strip().lower() == str(pred).strip().lower():
            correct += 1

    acc = (correct / total) * 100.0
    return {
        "accuracy": acc,
        "correct": correct,
        "total": total
    }
