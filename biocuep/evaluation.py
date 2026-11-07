"""
BioCUEP Intrinsic Evaluation & Error Analysis
Implements Section 5.3 (Intrinsic Evaluation) and Section 5.4 (Error Analysis) of the paper:
1. Five Intrinsic Metrics (Table 5):
   - NER Precision: valid biomedical term ratio
   - Sense Accuracy: correct UMLS sense disambiguation ratio
   - Question Coverage: fraction of questions having at least one core concept definition
   - Linking Accuracy: BKG node linking correctness ratio
   - Relevance Precision: factual utility of selected relations
2. Input Length vs Error Correlation (Section 5.4):
   - Pearson correlation between token length and prediction error
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_intrinsic_metrics(
    definitions_annotated: List[Dict[str, int]],
    relations_annotated: List[Dict[str, int]],
    num_questions: int
) -> Dict[str, float]:
    """
    Computes the 5 intrinsic evaluation metrics defined in Section 5.3:
    Each definition has:
      - N: valid biomedical term (0 or 1)
      - M: correct sense in context (0 or 1)
      - Q: key concept of question (0 or 1)
      - question_id: id of the associated question
    Each relation has:
      - L: KG node is intended concept (0 or 1)
      - R: relation relates to question and context (0 or 1)
    """
    # 1. NER Precision
    total_d = len(definitions_annotated)
    valid_n = sum(1 for d in definitions_annotated if d.get("N", 0) == 1)
    ner_precision = (valid_n / total_d * 100.0) if total_d > 0 else 0.0

    # 2. Sense Accuracy
    valid_m = sum(1 for d in definitions_annotated if d.get("N", 0) == 1 and d.get("M", 0) == 1)
    sense_accuracy = (valid_m / valid_n * 100.0) if valid_n > 0 else 0.0

    # 3. Question Coverage
    # Questions that have at least one d with N=1, M=1, Q=1
    covered_q_ids = set()
    for d in definitions_annotated:
        if d.get("N", 0) == 1 and d.get("M", 0) == 1 and d.get("Q", 0) == 1:
            q_id = d.get("question_id")
            if q_id is not None:
                covered_q_ids.add(q_id)
    question_coverage = (len(covered_q_ids) / num_questions * 100.0) if num_questions > 0 else 0.0

    # 4. Linking Accuracy
    total_r = len(relations_annotated)
    valid_l = sum(1 for r in relations_annotated if r.get("L", 0) == 1)
    linking_accuracy = (valid_l / total_r * 100.0) if total_r > 0 else 0.0

    # 5. Relevance Precision
    valid_r = sum(1 for r in relations_annotated if r.get("R", 0) == 1)
    relevance_precision = (valid_r / total_r * 100.0) if total_r > 0 else 0.0

    return {
        "NER_precision": ner_precision,
        "Sense_accuracy": sense_accuracy,
        "Question_coverage": question_coverage,
        "Linking_accuracy": linking_accuracy,
        "Relevance_precision": relevance_precision,
        "counts": {
            "total_definitions": total_d,
            "valid_NER": valid_n,
            "valid_Sense": valid_m,
            "covered_questions": len(covered_q_ids),
            "total_questions": num_questions,
            "total_relations": total_r,
            "valid_Link": valid_l,
            "valid_Rel": valid_r
        }
    }


def compute_length_error_correlation(
    token_lengths: List[int],
    is_correct_list: List[bool]
) -> Tuple[float, float]:
    """
    Computes Pearson correlation coefficient between input token length and prediction error
    (error = 1 if not correct else 0), as described in Section 5.4.
    Returns:
      (pearson_r, approximate_p_value)
    """
    n = len(token_lengths)
    if n < 3:
        return 0.0, 1.0

    errors = [0 if c else 1 for c in is_correct_list]

    mean_x = sum(token_lengths) / n
    mean_y = sum(errors) / n

    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(token_lengths, errors))
    var_x = sum((x - mean_x) ** 2 for x in token_lengths)
    var_y = sum((y - mean_y) ** 2 for y in errors)

    if var_x == 0 or var_y == 0:
        return 0.0, 1.0

    r = cov_xy / math.sqrt(var_x * var_y)

    # Standard t-statistic for Pearson correlation
    r_clamped = max(-0.999999, min(0.999999, r))
    t_stat = r_clamped * math.sqrt((n - 2) / (1 - r_clamped ** 2))
    # Approximate two-sided p-value
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2))))

    return r, p_val
