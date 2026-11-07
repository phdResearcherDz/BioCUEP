"""
Stage 6: Context Formatting & Atomic Truncation Assembly
Implements Section 3.6 and Section 4.3 of the paper:
1. Template-based formatting of selected relations and UMLS definitions (Figure 2).
2. Input assembly and atomic truncation:
   - Priority order: Question and original context first (kept first), then relations, then definitions.
   - Atomic Truncation Rule: A relation or definition that does not fit entirely is removed rather
     than cut, so that no incomplete fact is added.
   - Strict adherence to token budget (512 for fine-tuned readers, 8192 for LLMs).
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def format_relations_to_text(relations: List[Dict[str, str]]) -> Tuple[str, List[str]]:
    """
    Formats relation triplets into natural language sentences per Figure 2 & Section 3.6:
    '{source} has relations: {rel1} with {target1}, {rel2} with {target2}. '
    Returns:
      (full_relations_text, list_of_individual_relation_sentences)
    """
    if not relations:
        return "", []

    # Group by source node
    relations_dict: Dict[str, List[str]] = {}
    for r in relations:
        source = r.get("source", "").strip()
        rel_type = r.get("relation", "").strip()
        target = r.get("target_nodes", r.get("target", "")).strip()
        if not source or not target:
            continue
        phrase = f"{rel_type} with {target}"
        if source not in relations_dict:
            relations_dict[source] = []
        if phrase not in relations_dict[source]:
            relations_dict[source].append(phrase)

    sentences = []
    for source, phrases in relations_dict.items():
        sentences.append(f"{source} has relations: {', '.join(phrases)}.")

    full_text = " ".join(sentences)
    return full_text, sentences


def format_definitions_to_text(
    definitions_dict: Dict[str, str]
) -> Tuple[str, List[str]]:
    """
    Formats concept definitions into natural language sentences per Figure 2 & Section 3.6:
    '{concept} defined as following: {definition}. '
    Returns:
      (full_definitions_text, list_of_individual_definition_sentences)
    """
    if not definitions_dict:
        return "", []

    sentences = []
    for concept, defn in definitions_dict.items():
        c_clean = str(concept).strip()
        d_clean = str(defn).strip().rstrip(".")
        if c_clean and d_clean:
            sentences.append(f"{c_clean} defined as following: {d_clean}.")

    full_text = " ".join(sentences)
    return full_text, sentences


def extract_definitions_from_sample(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Extracts all UMLS definitions present in context_entities and question_entities.
    """
    defs: Dict[str, str] = {}
    for key in ("context_entities", "question_entities"):
        ents = sample.get(key, []) or []
        for ent_dict in ents:
            if not isinstance(ent_dict, dict):
                continue
            for mention, meta in ent_dict.items():
                if isinstance(meta, dict) and meta.get("definition"):
                    defs[mention] = meta["definition"]
    return defs


def count_tokens(text: str, tokenizer=None) -> int:
    """Helper to count tokens using HuggingFace tokenizer or word whitespace count."""
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.tokenize(text))
        except Exception:
            pass
    # Rough approximation: 1 token ~ 0.75 words, or word count + punctuation
    return int(len(text.split()) * 1.3) + 1


def assemble_enriched_input(
    question: str,
    context: str,
    relations: List[Dict[str, str]],
    definitions: Dict[str, str],
    tokenizer=None,
    max_tokens: int = 512,
    include_relations: bool = True,
    include_definitions: bool = True,
    sep_token: str = "[SEP]"
) -> Dict[str, Any]:
    """
    Assembles final model input adhering strictly to Section 4.3:
    1. Question and original context first (kept first).
    2. Then relations, added atomically without cutting facts.
    3. Then definitions, added atomically into remaining space.
    """
    q_text = question.strip()
    c_text = context.strip()

    base_text = f"{q_text} {c_text}"
    base_tokens = count_tokens(base_text, tokenizer)

    # If base exceeds max_tokens, truncate context (as in baseline)
    if base_tokens > max_tokens:
        words = c_text.split()
        while words and count_tokens(f"{q_text} {' '.join(words)}", tokenizer) > max_tokens:
            words.pop()
        c_text = " ".join(words)
        base_text = f"{q_text} {c_text}"
        base_tokens = count_tokens(base_text, tokenizer)

    remaining_budget = max(0, max_tokens - base_tokens)
    sep_cost = count_tokens(f" {sep_token} ", tokenizer)

    selected_relation_sentences: List[str] = []
    if include_relations and relations and remaining_budget > sep_cost:
        _, rel_sentences = format_relations_to_text(relations)
        # Prefix "Relations: "
        prefix_cost = count_tokens(" Relations: ", tokenizer)
        current_cost = sep_cost + prefix_cost
        
        for sent in rel_sentences:
            sent_cost = count_tokens(" " + sent, tokenizer)
            if current_cost + sent_cost <= remaining_budget:
                selected_relation_sentences.append(sent)
                current_cost += sent_cost
            else:
                # Rule: Atomic omission rather than cutting mid-sentence
                break
        
        if selected_relation_sentences:
            remaining_budget -= current_cost

    selected_definition_sentences: List[str] = []
    if include_definitions and definitions and remaining_budget > 0:
        _, def_sentences = format_definitions_to_text(definitions)
        prefix_cost = count_tokens(" Definitions: ", tokenizer)
        current_cost = prefix_cost

        for sent in def_sentences:
            sent_cost = count_tokens(" " + sent, tokenizer)
            if current_cost + sent_cost <= remaining_budget:
                selected_definition_sentences.append(sent)
                current_cost += sent_cost
            else:
                # Rule: Atomic omission rather than cutting mid-sentence
                break

    # Build final metadata block
    metadata_parts = []
    if selected_relation_sentences:
        metadata_parts.append(f"Relations: {' '.join(selected_relation_sentences)}")
    if selected_definition_sentences:
        metadata_parts.append(f"Definitions: {' '.join(selected_definition_sentences)}")

    metadata_text = " ".join(metadata_parts).strip()

    if metadata_text:
        final_input = f"{q_text} {c_text} {sep_token} {metadata_text}"
    else:
        final_input = f"{q_text} {c_text}"

    return {
        "enriched_input": final_input,
        "metadata": metadata_text,
        "relations_text": " ".join(selected_relation_sentences),
        "definitions_text": " ".join(selected_definition_sentences),
        "total_tokens": count_tokens(final_input, tokenizer),
        "num_relations_kept": len(selected_relation_sentences),
        "num_definitions_kept": len(selected_definition_sentences)
    }
