"""
Stage 5: Semantic Relation Filtering
Implements Section 3.5, Equations (7) and (8) of the paper:
1. Entity-link / Question-Context (QC) Filtering:
   RSQ = {(h, r, t) in RS U RQ | h, t in LS U LQ} (Equation 7)
2. Similarity-based Filtering:
   Rsim = Top-k(RS U RQ, Sim(vQ, vr)) using dense retrieval encoder TAS-B (Equation 8)
   where k = 10 relations per KG (Section 4.3).
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Cached TAS-B model and tokenizer
_TASB_TOKENIZER = None
_TASB_MODEL = None


def get_all_linked_concepts_set(
    sample: Dict[str, Any],
    kg_name: str
) -> Set[str]:
    """
    Extracts LS U LQ (all linked concepts in context and question) as a set.
    Includes both the mapped BKG node names and normalized mentions.
    """
    ctx_concepts = sample.get(f"{kg_name}_concepts_context", {}) or {}
    q_concepts = sample.get(f"{kg_name}_concepts_question", {}) or {}

    concepts_set: Set[str] = set()
    # Add BKG node names (values)
    for v in ctx_concepts.values():
        if v:
            concepts_set.add(v.strip().lower())
    for v in q_concepts.values():
        if v:
            concepts_set.add(v.strip().lower())
    # Add mentions (keys) for robustness
    for k in ctx_concepts.keys():
        if k:
            concepts_set.add(k.strip().lower())
    for k in q_concepts.keys():
        if k:
            concepts_set.add(k.strip().lower())

    return concepts_set


def collect_all_triples(sample: Dict[str, Any], kg_name: str) -> List[Dict[str, str]]:
    """
    Aggregates all first-level triples from RS and RQ for the sample.
    """
    q_data = sample.get(f"{kg_name}_concepts_question_with_kg_data", {}) or {}
    ctx_data = sample.get(f"{kg_name}_concepts_context_with_kg_data", {}) or {}

    seen_triples = set()
    unique_triples = []

    for group in (q_data.values(), ctx_data.values()):
        for concept_entry in group:
            if not isinstance(concept_entry, dict):
                continue
            for triple in concept_entry.get("triples", []):
                h = str(triple.get("source", "")).strip()
                r = str(triple.get("relation", "")).strip()
                t = str(triple.get("target_nodes", triple.get("target", ""))).strip()
                key = (h.lower(), r.lower(), t.lower())
                if key not in seen_triples and h and t:
                    seen_triples.add(key)
                    unique_triples.append({"source": h, "relation": r, "target_nodes": t})

    return unique_triples


def filter_relations_qc(sample: Dict[str, Any], kg_name: str) -> List[Dict[str, str]]:
    """
    Implements Entity-Link Filtering (QC Filter) per Equation (7):
    RSQ = {(h, r, t) in RS U RQ | h, t in LS U LQ}
    Retains only relations that directly connect concepts found in both context and question.
    """
    linked_concepts = get_all_linked_concepts_set(sample, kg_name)
    all_triples = collect_all_triples(sample, kg_name)

    qc_filtered = []
    for triple in all_triples:
        h = triple["source"].strip().lower()
        t = triple["target_nodes"].strip().lower()

        # Both head and tail must be linked concepts from LS U LQ
        if h in linked_concepts and t in linked_concepts:
            qc_filtered.append(triple)

    return qc_filtered


def load_tasb_model(device_str: Optional[str] = None):
    """
    Loads TAS-B dense retrieval encoder (sentence-transformers/msmarco-distilbert-base-tas-b).
    """
    global _TASB_TOKENIZER, _TASB_MODEL
    if _TASB_TOKENIZER is not None and _TASB_MODEL is not None:
        return _TASB_TOKENIZER, _TASB_MODEL

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        from biocuep.config import TAS_B_MODEL
        _TASB_TOKENIZER = AutoTokenizer.from_pretrained(TAS_B_MODEL)
        _TASB_MODEL = AutoModel.from_pretrained(TAS_B_MODEL)
        device = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        _TASB_MODEL.to(device)
        _TASB_MODEL.eval()
        return _TASB_TOKENIZER, _TASB_MODEL
    except Exception as e:
        logger.warning(f"Could not load TAS-B model: {e}. Lexical scoring fallback will be used.")
        return None, None


def filter_relations_similarity(
    sample: Dict[str, Any],
    kg_name: str,
    top_k: int = 10,
    tokenizer=None,
    model=None,
    device: str = "cpu"
) -> List[Dict[str, str]]:
    """
    Implements Similarity-based Filtering per Equation (8):
    Rsim = Top-k(RS U RQ, Sim(vQ, vr))
    Scores relations using dot-product similarity between TAS-B question embedding vQ and relation text vr.
    """
    question = sample.get("question", "")
    all_triples = collect_all_triples(sample, kg_name)

    if not all_triples:
        return []

    if len(all_triples) <= top_k:
        return all_triples

    if tokenizer is not None and model is not None:
        try:
            import torch
            triple_texts = [f"{t['source']} {t['relation']} {t['target_nodes']}" for t in all_triples]

            # Encode question
            q_inputs = tokenizer([question], padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                q_emb = model(**q_inputs).last_hidden_state[:, 0]  # CLS pooling

            # Encode relations in batch
            r_inputs = tokenizer(triple_texts, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                r_emb = model(**r_inputs).last_hidden_state[:, 0]

            # Dot-product similarity
            scores = torch.mm(q_emb, r_emb.transpose(0, 1))[0].cpu().tolist()
            scored_triples = sorted(zip(all_triples, scores), key=lambda x: x[1], reverse=True)
            return [t for t, s in scored_triples[:top_k]]
        except Exception as e:
            logger.debug(f"TAS-B dense scoring failed: {e}. Falling back to lexical overlap.")

    # Fallback lexical scoring: overlap with question terms
    q_tokens = set(question.lower().split())
    scored = []
    for t in all_triples:
        text = f"{t['source']} {t['relation']} {t['target_nodes']}".lower()
        score = sum(1 for tok in text.split() if tok in q_tokens)
        scored.append((t, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, s in scored[:top_k]]
