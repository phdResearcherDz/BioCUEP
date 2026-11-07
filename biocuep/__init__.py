"""
BioCUEP: Biomedical Context Understanding Enhancement Pipeline
Accompanying codebase for the paper:
'Enhancing Biomedical Context Understanding in Language Models using Medical Knowledge Graphs: Application to Question Answering Systems'

Main components:
- extraction: Multi-NER biomedical entity extraction (SciSpaCy CRAFT, JNLPBA, BC5CDR, BIONLP13CG)
- normalization: UMLS candidate retrieval + BioBERT 10-word sliding window disambiguation
- linking: CODER embedding similarity linking to BKGs (Hetionet, PrimeKG) with tau = 0.90
- relations: First-level relation extraction from BKGs
- filtering: Entity-link (QC) filtering and TAS-B dense similarity filtering
- formatting: Template-based formatting with atomic non-cutting truncation
- baselines: RRI, Top-N, Cluster-based, TF-IDF-RI, PBCA, CoT
- experiments: 14 ablation study configurations (Table 2)
- readers: Downstream fine-tuned & few-shot reader models
- evaluation: Intrinsic evaluation metrics (Table 5) & error correlation
"""

__version__ = "1.0.0"
__pipeline__ = "BioCUEP"

from biocuep.config import BioCUEPConfig
from biocuep.pipeline import BioCUEPPipeline
from biocuep.experiments import get_experiment_config, EXPERIMENT_CONFIGS
