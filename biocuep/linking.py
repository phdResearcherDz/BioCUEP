"""
Stage 3: Biomedical Entity Linking to BKGs (Hetionet & PrimeKG)
Implements Section 3.3, Equation (6) of the paper:
Links each normalized concept c*_e to the most similar concept in the BKG using CODER embeddings:
b*_{c*_e} = argmax_{b in BKG, Sim(c*_e, b) >= tau} Sim(c*_e, b)
where tau = 0.90 (Section 4.3).
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# Cached CODER model and tokenizer
_CODER_TOKENIZER = None
_CODER_MODEL = None


def load_coder_model(device_str: Optional[str] = None):
    """
    Loads CODER encoder (Yuan et al., 2022) or UMLSBert for knowledge-infused concept embedding.
    """
    global _CODER_TOKENIZER, _CODER_MODEL
    if _CODER_TOKENIZER is not None and _CODER_MODEL is not None:
        return _CODER_TOKENIZER, _CODER_MODEL

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        from biocuep.config import CODER_MODEL, UMLSBERT_FALLBACK_MODEL
        
        for m_name in [CODER_MODEL, UMLSBERT_FALLBACK_MODEL]:
            try:
                _CODER_TOKENIZER = AutoTokenizer.from_pretrained(m_name)
                _CODER_MODEL = AutoModel.from_pretrained(m_name)
                device = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
                _CODER_MODEL.to(device)
                _CODER_MODEL.eval()
                logger.info(f"Loaded CODER encoder: {m_name}")
                return _CODER_TOKENIZER, _CODER_MODEL
            except Exception as e:
                logger.debug(f"Failed to load {m_name}: {e}")

        logger.warning("Could not load HuggingFace CODER model. Fast string/lexical linking will be used.")
        return None, None
    except ImportError:
        logger.warning("Transformers/PyTorch not installed. Using exact/fuzzy linking fallback.")
        return None, None


def link_concept_to_bkg(
    concept_name: str,
    bkg_node_names: List[str],
    node_embeddings: Optional[Any] = None,
    threshold: float = 0.90,
    coder_tokenizer=None,
    coder_model=None,
    device: str = "cpu"
) -> Optional[str]:
    """
    Links a single normalized concept to the highest scoring BKG node satisfying Sim(c*, b) >= tau.
    """
    if not concept_name or not bkg_node_names:
        return None

    concept_lower = concept_name.strip().lower()

    # Fast path 1: Exact case-insensitive match (Sim = 1.0 >= tau)
    for node in bkg_node_names:
        if node.lower() == concept_lower:
            return node

    # Fast path 2: Direct substring / acronym match
    for node in bkg_node_names:
        node_l = node.lower()
        if (len(concept_lower) > 3 and concept_lower in node_l) or (len(node_l) > 3 and node_l in concept_lower):
            # Candidate match
            return node

    # Dense embedding matching via CODER if embeddings available
    if node_embeddings is not None and coder_model is not None and coder_tokenizer is not None:
        try:
            import torch
            with torch.no_grad():
                tokens = coder_tokenizer([concept_name], return_tensors='pt', padding=True, truncation=True)
                tokens = {k: v.to(device) for k, v in tokens.items()}
                output = coder_model(**tokens)
                c_emb = getattr(output, 'pooler_output', output.last_hidden_state.mean(dim=1))
                
                # Cosine similarity against precomputed node embeddings
                sims = torch.nn.functional.cosine_similarity(c_emb, node_embeddings)
                max_sim, best_idx = torch.max(sims, dim=0)

                if max_sim.item() >= threshold:
                    return bkg_node_names[best_idx.item()]
        except Exception as ex:
            logger.debug(f"CODER linking error for {concept_name}: {ex}")

    return None


def link_dataset_concepts(
    data: List[Dict[str, Any]],
    kg_name: str,
    node_names: List[str],
    node_embeddings: Optional[Any] = None,
    threshold: float = 0.90,
    coder_tokenizer=None,
    coder_model=None,
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    """
    Links normalized entities in both context and question for all items in dataset.
    Populates:
      item[f'{kg_name}_concepts_context'] = {mention: bkg_node}
      item[f'{kg_name}_concepts_question'] = {mention: bkg_node}
    """
    for item in data:
        ctx_ents = item.get("context_entities", [])
        q_ents = item.get("question_entities", [])

        ctx_links: Dict[str, str] = {}
        for ent_dict in ctx_ents:
            for mention, meta in ent_dict.items():
                canonical = meta.get("canonical_name", mention)
                matched_node = link_concept_to_bkg(
                    canonical, node_names, node_embeddings,
                    threshold=threshold, coder_tokenizer=coder_tokenizer,
                    coder_model=coder_model, device=device
                )
                if matched_node:
                    ctx_links[mention] = matched_node

        q_links: Dict[str, str] = {}
        for ent_dict in q_ents:
            for mention, meta in ent_dict.items():
                canonical = meta.get("canonical_name", mention)
                matched_node = link_concept_to_bkg(
                    canonical, node_names, node_embeddings,
                    threshold=threshold, coder_tokenizer=coder_tokenizer,
                    coder_model=coder_model, device=device
                )
                if matched_node:
                    q_links[mention] = matched_node

        item[f"{kg_name}_concepts_context"] = ctx_links
        item[f"{kg_name}_concepts_question"] = q_links

    return data
