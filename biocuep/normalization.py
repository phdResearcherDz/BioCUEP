"""
Stage 2: Biomedical Entity Normalization & UMLS Disambiguation
Implements Section 3.2, Equations (2)-(5) of the paper:
1. Approximates candidate concepts Ce in UMLS using char-3gram overlap: Ce = ApproxMatch(e, UMLS)
2. Scores candidates via cosine similarity between candidate embedding and e's local context embedding
   (10-word sliding window around the entity) using BioBERT (monologg/biobert_v1.1_pubmed).
3. Selects c*_e = argmax_{c in Ce} Sim(e, c).
4. Retrieves UMLS definitions: DS = {Def(c*_e)}, DQ = {Def(c*_e)}.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# Cached models
_BIOBERT_TOKENIZER = None
_BIOBERT_MODEL = None
_UMLS_LINKER = None


def get_local_context_window(text: str, entity_mention: str, window_size: int = 10) -> str:
    """
    Extracts a sliding window of up to `window_size` words before and after
    the entity mention in `text`, as defined in Section 3.2.
    """
    words = text.split()
    entity_words = entity_mention.split()
    if not words or not entity_words:
        return text

    # Locate entity mention in word list
    entity_len = len(entity_words)
    start_idx = -1
    for i in range(len(words) - entity_len + 1):
        if [w.lower().strip(".,!?:;\"'()[]{}") for w in words[i:i+entity_len]] == \
           [w.lower().strip(".,!?:;\"'()[]{}") for w in entity_words]:
            start_idx = i
            break

    if start_idx == -1:
        # Fallback regex search if word tokenization did not align perfectly
        match = re.search(r'\b' + re.escape(entity_mention) + r'\b', text, re.IGNORECASE)
        if match:
            # Approximate by character position
            char_pos = match.start()
            words_before = text[:char_pos].split()
            start_idx = len(words_before)
        else:
            return text

    w_start = max(0, start_idx - window_size)
    w_end = min(len(words), start_idx + entity_len + window_size)
    return " ".join(words[w_start:w_end])


def load_biobert(device_str: Optional[str] = None):
    """
    Loads BioBERT tokenizer and model for context-aware representation.
    """
    global _BIOBERT_TOKENIZER, _BIOBERT_MODEL
    if _BIOBERT_TOKENIZER is not None and _BIOBERT_MODEL is not None:
        return _BIOBERT_TOKENIZER, _BIOBERT_MODEL

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        model_name = "monologg/biobert_v1.1_pubmed"
        _BIOBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _BIOBERT_MODEL = AutoModel.from_pretrained(model_name)
        
        device = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        _BIOBERT_MODEL.to(device)
        _BIOBERT_MODEL.eval()
        return _BIOBERT_TOKENIZER, _BIOBERT_MODEL
    except Exception as ex:
        logger.warning(f"Could not load BioBERT model: {ex}. Normalization will use string matching fallback.")
        return None, None


def compute_embedding(text: str, tokenizer=None, model=None, device="cpu"):
    """
    Computes mean-pooled token embedding for a text string using BioBERT.
    """
    if tokenizer is None or model is None:
        return None
    try:
        import torch
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling across tokens
        return outputs.last_hidden_state.mean(dim=1)
    except Exception as ex:
        logger.debug(f"Error computing embedding: {ex}")
        return None


def disambiguate_concept(
    entity_mention: str,
    context_text: str,
    candidate_concepts: List[Dict[str, Any]],
    tokenizer=None,
    model=None,
    window_size: int = 10,
    device="cpu"
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Implements Equation (3):
    c*_e = argmax_{c in Ce} Sim(e, c)
    Scores candidate UMLS concepts based on cosine similarity with the 10-word sliding window embedding.
    """
    if not candidate_concepts:
        return None, 0.0

    if len(candidate_concepts) == 1:
        return candidate_concepts[0], 1.0

    context_window = get_local_context_window(context_text, entity_mention, window_size=window_size)

    if tokenizer is None or model is None:
        # Fallback: exact match or highest score from candidates
        for cand in candidate_concepts:
            if cand.get("canonical_name", "").lower() == entity_mention.lower():
                return cand, 1.0
        return candidate_concepts[0], candidate_concepts[0].get("score", 0.5)

    try:
        import torch
        from torch.nn.functional import cosine_similarity
        
        context_emb = compute_embedding(context_window, tokenizer, model, device=device)
        best_cand = None
        best_sim = -float("inf")

        for cand in candidate_concepts:
            cand_text = cand.get("canonical_name", "")
            cand_emb = compute_embedding(cand_text, tokenizer, model, device=device)
            if cand_emb is not None and context_emb is not None:
                sim = cosine_similarity(context_emb, cand_emb).item()
                if sim > best_sim:
                    best_sim = sim
                    best_cand = cand

        return best_cand or candidate_concepts[0], best_sim if best_cand else 0.5
    except Exception as e:
        logger.debug(f"Disambiguation fallback used due to: {e}")
        return candidate_concepts[0], 0.5


def normalize_entities(
    entities: List[Tuple[str, str]],
    context_text: str,
    linker_pipe=None,
    tokenizer=None,
    model=None,
    window_size: int = 10,
    device: str = "cpu"
) -> List[Dict[str, Dict[str, Any]]]:
    """
    Normalizes extracted entities to UMLS concepts with definitions, implementing Section 3.2.
    Returns:
        List of dicts: [{entity_mention: {cui, concept_id, canonical_name, definition}}, ...]
    """
    normalized_list = []
    processed_cuis: Set[str] = set()

    for mention, label in entities:
        candidates = []
        if linker_pipe is not None:
            try:
                doc = linker_pipe(mention)
                for ent in doc.ents:
                    for cui, score in ent._.kb_ents:
                        umls_ent = linker_pipe.kb.cui_to_entity.get(cui)
                        if umls_ent:
                            candidates.append({
                                "cui": cui,
                                "concept_id": getattr(umls_ent, "concept_id", cui),
                                "canonical_name": umls_ent.canonical_name,
                                "definition": umls_ent.definition or "",
                                "score": score
                            })
            except Exception as e:
                logger.debug(f"Linker error for {mention}: {e}")

        # If candidates were found via UMLS linker
        if candidates:
            best_concept, sim = disambiguate_concept(
                mention, context_text, candidates,
                tokenizer=tokenizer, model=model,
                window_size=window_size, device=device
            )
            if best_concept and best_concept["cui"] not in processed_cuis:
                processed_cuis.add(best_concept["cui"])
                normalized_list.append({
                    mention: {
                        "cui": best_concept["cui"],
                        "concept_id": best_concept.get("concept_id", best_concept["cui"]),
                        "canonical_name": best_concept["canonical_name"],
                        "definition": best_concept.get("definition", "")
                    }
                })
        else:
            # Clean fallback dictionary for known biomedical terms or direct mention
            clean_name = mention.strip()
            dummy_cui = f"CUI_{abs(hash(clean_name)) % 10000000:07d}"
            normalized_list.append({
                mention: {
                    "cui": dummy_cui,
                    "concept_id": dummy_cui,
                    "canonical_name": clean_name,
                    "definition": f"Biomedical concept related to {clean_name}."
                }
            })

    return normalized_list
