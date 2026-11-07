"""
Stage 1: Biomedical Entity Extraction
Implements Section 3.1, Equation (1) of the paper:
Extracts biomedical entities from question Q and context segments S using 4 SciSpaCy NER models
(CRAFT, JNLPBA, BC5CDR, BIONLP13CG), applies post-extraction filtering F, and deduplicates to form ES and EQ.
"""

import re
import logging
from typing import List, Set, Tuple, Dict, Any, Optional
from biocuep.config import RELEVANT_LABELS, NER_MODELS

logger = logging.getLogger(__name__)

# Cache for loaded spaCy NER models
_LOADED_NER_MODELS = {}


def is_full_word(context: str, entity: str) -> bool:
    """
    Check if the entity appears as a discrete whole word/term in context,
    preventing noisy partial token matches.
    """
    if not entity or not context:
        return False
    pattern = r'\b' + re.escape(entity.strip()) + r'\b'
    return re.search(pattern, context, flags=re.IGNORECASE) is not None


def filtering_function(entity_text: str, entity_label: Optional[str] = None) -> bool:
    """
    Filtering function F (Section 3.1):
    Removes noisy, truncated, single-letter, or irrelevant terms.
    """
    clean_text = entity_text.strip()
    if len(clean_text) <= 1:
        return False
    # Discard pure numbers or punctuation
    if not any(char.isalpha() for char in clean_text):
        return False
    # If label is provided, ensure it falls within biomedical relevant types
    if entity_label and entity_label not in RELEVANT_LABELS:
        return False
    return True


def get_ner_models():
    """
    Lazily loads the 4 SciSpaCy models. If spaCy/models are not installed,
    returns an empty dict or logs a warning.
    """
    global _LOADED_NER_MODELS
    if _LOADED_NER_MODELS:
        return _LOADED_NER_MODELS

    try:
        import spacy
        for key, model_name in NER_MODELS.items():
            try:
                _LOADED_NER_MODELS[key] = spacy.load(model_name)
            except Exception as e:
                logger.warning(f"Could not load SciSpaCy model '{model_name}': {e}. Ensure it is installed via pip.")
    except ImportError:
        logger.warning("spaCy is not installed. Multi-model NER will require rule-based fallback.")

    return _LOADED_NER_MODELS


def extract_entities_from_text(
    text: str,
    models: Optional[Dict[str, Any]] = None,
    relevant_labels: Set[str] = RELEVANT_LABELS
) -> List[Tuple[str, str]]:
    """
    Extracts entities from a single text segment using the ensemble of 4 NER models.
    Applies filtering function F and returns deduplicated list of (entity_text, entity_label).
    """
    if not text or not text.strip():
        return []

    if models is None:
        models = get_ner_models()

    extracted_set: Set[Tuple[str, str]] = set()

    if models:
        for model_name, nlp in models.items():
            try:
                doc = nlp(text)
                for ent in doc.ents:
                    ent_text = ent.text.strip()
                    if is_full_word(text, ent_text) and filtering_function(ent_text, ent.label_):
                        extracted_set.add((ent_text, ent.label_))
            except Exception as ex:
                logger.debug(f"Error processing text with model {model_name}: {ex}")
    else:
        # Fallback heuristic extractor when SciSpaCy models are not yet downloaded in the local environment
        # Matches capitalized terms, hyphenated biomedical terms, gene/protein codes, chemical suffixes
        candidates = re.findall(r'\b[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)+\b|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b|\b[A-Z]{2,}[0-9]*\b|\b[a-z]+(?:ase|in|one|ide|ol|ine|ic|gen|itis)\b', text, flags=re.IGNORECASE)
        for cand in candidates:
            if filtering_function(cand):
                extracted_set.add((cand, "BIOMEDICAL_CONCEPT"))

    return sorted(list(extracted_set), key=lambda x: x[0].lower())


def extract_entities_for_sample(
    question: str,
    context: str,
    models: Optional[Dict[str, Any]] = None
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Extracts EQ (Question entities) and ES (Context entities) per Equation (1):
    ES = Union_{j=1..4} (Union_{i=1..N} F(Mj(Si)))
    EQ = Union_{j=1..4} F(Mj(Q))
    """
    eq = extract_entities_from_text(question, models=models)
    es = extract_entities_from_text(context, models=models)
    return eq, es
