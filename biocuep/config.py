"""
BioCUEP Configuration Module
Centralized configuration parameters for the Biomedical Context Understanding Enhancement Pipeline (BioCUEP).
Paper: 'Enhancing Biomedical Context Understanding in Language Models using Medical Knowledge Graphs: Application to Question Answering Systems'
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSED_DATA_DIR = PROJECT_ROOT / "Pre_Processed_Datasets"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Supported Benchmarks (Section 4.1)
SUPPORTED_DATASETS = [
    "BioASQ_BLURB",   # BioASQ 7B from BLURB benchmark (Yes/No)
    "BioASQ2024",     # BioASQ 12B (2024 release) (Yes/No)
    "PubmedQA",       # PubMedQA BLURB annotated split (Yes/No/Maybe)
]

# Random seeds used in experiments (Section 4.3 & README)
DEFAULT_SEEDS = [7, 42, 123, 2024, 31415]
DEFAULT_SEED = 42

# Hardware / Device configuration
DEFAULT_DEVICE = "cuda"  # fallback to 'cpu' if unavailable

# Stage 1: Biomedical Entity Extraction Models (SciSpaCy - Section 3.1)
NER_MODELS = {
    "craft": "en_ner_craft_md",       # Multi-type concepts (Bada et al., 2012)
    "jnlpba": "en_ner_jnlpba_md",     # Genes/proteins (Huang et al., 2019)
    "bc5cdr": "en_ner_bc5cdr_md",     # Chemicals & diseases (Li et al., 2016)
    "bionlp13cg": "en_ner_bionlp13cg_md" # Anatomy, cell types, cancer (Pyysalo et al., 2013)
}

# Relevant biomedical entity labels for filtering (Section 3.1)
RELEVANT_LABELS = {
    "AMINO_ACID", "ANATOMICAL_SYSTEM", "CANCER", "CELL", "CELLULAR_COMPONENT",
    "GENE_OR_GENE_PRODUCT", "IMMATERIAL_ANATOMICAL_ENTITY",
    "ORGAN", "PATHOLOGICAL_FORMATION", "SIMPLE_CHEMICAL", "TISSUE",
    "DISEASE", "CHEMICAL", "DNA", "CELL_TYPE", "CELL_LINE", "RNA", "PROTEIN",
    "GGP", "SO", "TAXON", "CHEBI", "GO", "CL"
}

# Stage 2: Entity Normalization & UMLS Disambiguation (Section 3.2)
UMLS_SCIBERT_MODEL = "en_core_sci_scibert"
NORMALIZATION_EMBEDDING_MODEL = "monologg/biobert_v1.1_pubmed"
NORMALIZATION_WINDOW_SIZE = 10  # 10-word sliding window around entity (Section 3.2)
UMLS_LINKER_THRESHOLD = 0.95
UMLS_MAX_ENTITIES_PER_MENTION = 5

# Stage 3: Entity Linking to BKG (Section 3.3 & Section 4.3)
CODER_MODEL = "GanjinZero/coder_eng"  # CODER encoder (Yuan et al., 2022)
UMLSBERT_FALLBACK_MODEL = "GanjinZero/UMLSBert_ENG"
LINKING_SIMILARITY_THRESHOLD = 0.90  # tau = 0.90 (Section 4.3)

# Stage 4: Biomedical Knowledge Graphs (Section 4.2)
HETIONET_NEO4J_URI = os.getenv("HETIONET_NEO4J_URI", "bolt://neo4j.het.io:7687")
HETIONET_NEO4J_USER = os.getenv("HETIONET_NEO4J_USER", "neo4j")
HETIONET_NEO4J_PASSWORD = os.getenv("HETIONET_NEO4J_PASSWORD", "")

PRIMEKG_NEO4J_URI = os.getenv("PRIMEKG_NEO4J_URI", "bolt://localhost:7687")
PRIMEKG_NEO4J_USER = os.getenv("PRIMEKG_NEO4J_USER", "neo4j")
PRIMEKG_NEO4J_PASSWORD = os.getenv("PRIMEKG_NEO4J_PASSWORD", "")

# Stage 5: Relation Filtering (Section 3.5 & Section 4.3)
TAS_B_MODEL = "sentence-transformers/msmarco-distilbert-base-tas-b"  # Dense retrieval encoder
SIMILARITY_FILTER_TOP_K = 10  # k = 10 relations per KG (Section 4.3)

# Stage 6: Data Formatting & Token Budget (Section 3.6 & Section 4.3)
FINE_TUNED_READER_MAX_TOKENS = 512    # 512-token budget for BioLinkBERT / GPT-2 / T5
LLM_READER_MAX_TOKENS = 8192          # 8192-token budget for few-shot LLMs
DEFAULT_FEW_SHOT_TEMPERATURE = 0.1    # Temperature 0.1 for few-shot readers (Section 4.3)
DEFAULT_NUM_DEMONSTRATIONS = 3        # 3 fixed demonstrations (Section 4.3)

# Downstream models
BIOLINKBERT_BASE = "michiyasunaga/BioLinkBERT-base"
BIOLINKBERT_LARGE = "michiyasunaga/BioLinkBERT-large"
GPT2_MODEL = "gpt2"
T5_MODEL = "t5-base"

@dataclass
class BioCUEPConfig:
    """Config container allowing override of parameters per run."""
    dataset_name: str = "BioASQ_BLURB"
    data_dir: Path = PREPROCESSED_DATA_DIR
    linking_threshold: float = LINKING_SIMILARITY_THRESHOLD
    filter_top_k: int = SIMILARITY_FILTER_TOP_K
    window_size: int = NORMALIZATION_WINDOW_SIZE
    max_tokens: int = FINE_TUNED_READER_MAX_TOKENS
    device: str = DEFAULT_DEVICE
    seed: int = DEFAULT_SEED
    enable_gpu: bool = True
