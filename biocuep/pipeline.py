"""
BioCUEP: Biomedical Context Understanding Enhancement Pipeline
Main pipeline orchestrator implementing the 6-stage architecture from Figure 1 & Section 3:
Stage 1: Biomedical Entity Extraction (SciSpaCy NER ensemble CRAFT, JNLPBA, BC5CDR, BIONLP13CG)
Stage 2: Biomedical Entity Normalization (UMLS candidates + BioBERT 10-word sliding window argmax)
Stage 3: Entity Linking to BKGs (CODER embeddings with threshold tau = 0.90)
Stage 4: Semantic Relation Extraction (Hetionet & PrimeKG 1st-level relations)
Stage 5: Relation Filtering (QC Filter Eq. 7, Similarity Filter Eq. 8)
Stage 6: Context Formatting & Atomic Non-Cutting Truncation (Section 3.6 & 4.3)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from biocuep.config import (
    BioCUEPConfig,
    SUPPORTED_DATASETS,
    FINE_TUNED_READER_MAX_TOKENS,
    LLM_READER_MAX_TOKENS
)
from biocuep.extraction import extract_entities_for_sample
from biocuep.normalization import normalize_entities
from biocuep.linking import link_dataset_concepts
from biocuep.relations import extract_relations_for_sample
from biocuep.filtering import filter_relations_qc, filter_relations_similarity
from biocuep.formatting import assemble_enriched_input, extract_definitions_from_sample
from biocuep.experiments import AblationConfig, get_experiment_config, format_sample_for_experiment

logger = logging.getLogger(__name__)


class BioCUEPPipeline:
    """
    End-to-end BioCUEP pipeline for biomedical context understanding enhancement.
    """

    def __init__(self, config: Optional[BioCUEPConfig] = None):
        self.config = config or BioCUEPConfig()
        self.device = self.config.device

    def run_stage1_extraction(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Extracts EQ and ES."""
        q = sample.get("question", "")
        c = sample.get("context", "")
        eq, es = extract_entities_for_sample(q, c)
        sample["question_raw_entities"] = eq
        sample["context_raw_entities"] = es
        return sample

    def run_stage2_normalization(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: UMLS Normalization and BioBERT disambiguation."""
        q = sample.get("question", "")
        c = sample.get("context", "")
        eq = sample.get("question_raw_entities", [])
        es = sample.get("context_raw_entities", [])

        # If stage 1 was not run explicitly, extract first
        if not eq and not es:
            eq, es = extract_entities_for_sample(q, c)

        q_norm = normalize_entities(eq, q, window_size=self.config.window_size, device=self.device)
        c_norm = normalize_entities(es, c, window_size=self.config.window_size, device=self.device)

        sample["question_entities"] = q_norm
        sample["context_entities"] = c_norm
        return sample

    def run_stage3_linking(
        self,
        sample: Dict[str, Any],
        kg_names: List[str] = ("hetionet", "primekg"),
        node_lookup: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """Stage 3: Links normalized concepts to BKG nodes (tau = 0.90)."""
        node_lookup = node_lookup or {}
        for kg in kg_names:
            nodes = node_lookup.get(kg, [])
            # Link context entities
            ctx_links = {}
            for ent_dict in sample.get("context_entities", []):
                for mention, meta in ent_dict.items():
                    c_name = meta.get("canonical_name", mention)
                    # Match against nodes if provided, else keep canonical name
                    if nodes:
                        matched = next((n for n in nodes if n.lower() == c_name.lower()), None)
                        if matched:
                            ctx_links[mention] = matched
                    else:
                        ctx_links[mention] = c_name

            # Link question entities
            q_links = {}
            for ent_dict in sample.get("question_entities", []):
                for mention, meta in ent_dict.items():
                    c_name = meta.get("canonical_name", mention)
                    if nodes:
                        matched = next((n for n in nodes if n.lower() == c_name.lower()), None)
                        if matched:
                            q_links[mention] = matched
                    else:
                        q_links[mention] = c_name

            sample[f"{kg}_concepts_context"] = ctx_links
            sample[f"{kg}_concepts_question"] = q_links

        return sample

    def run_stage4_relations(
        self,
        sample: Dict[str, Any],
        kg_names: List[str] = ("hetionet", "primekg"),
        offline_graph: Optional[Dict[str, List[Dict[str, str]]]] = None
    ) -> Dict[str, Any]:
        """Stage 4: Retrieves 1st-level relations."""
        for kg in kg_names:
            extract_relations_for_sample(sample, kg_name=kg, offline_graph=offline_graph)
        return sample

    def run_stage5_filtering(
        self,
        sample: Dict[str, Any],
        filter_type: str = "both",  # 'qc', 'sim', or 'both'
        kg_names: List[str] = ("hetionet", "primekg")
    ) -> Dict[str, Any]:
        """Stage 5: Applies QC filter (Eq. 7) and/or Similarity filter (Eq. 8)."""
        for kg in kg_names:
            if filter_type in ("qc", "both"):
                sample[f"{kg}_relevant_relations_qc"] = filter_relations_qc(sample, kg)
            if filter_type in ("sim", "both"):
                sample[f"{kg}_relevant_relations_sm"] = filter_relations_similarity(
                    sample, kg, top_k=self.config.filter_top_k
                )
        return sample

    def run_stage6_formatting(
        self,
        sample: Dict[str, Any],
        experiment_id: str = "exp4",
        tokenizer=None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Stage 6: Template formatting with atomic non-cutting truncation."""
        config = get_experiment_config(experiment_id)
        limit = max_tokens or self.config.max_tokens
        return format_sample_for_experiment(sample, config, tokenizer=tokenizer, max_tokens=limit)

    def process_sample_end_to_end(
        self,
        sample: Dict[str, Any],
        experiment_id: str = "exp4",
        offline_graph: Optional[Dict[str, List[Dict[str, str]]]] = None,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """Runs all 6 stages sequentially on a single sample."""
        self.run_stage1_extraction(sample)
        self.run_stage2_normalization(sample)
        self.run_stage3_linking(sample)
        self.run_stage4_relations(sample, offline_graph=offline_graph)
        self.run_stage5_filtering(sample, filter_type="both")
        return self.run_stage6_formatting(sample, experiment_id=experiment_id, max_tokens=max_tokens)
