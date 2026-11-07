"""
BioCUEP Unified Execution Pipeline
Accompanying: "Enhancing Biomedical Context Understanding in Language Models using Medical Knowledge Graphs: Application to Question Answering Systems"

Executes the 6 pipeline stages sequentially or individually:
  Stage 1: Biomedical Entity Extraction (SciSpaCy NER CRAFT, JNLPBA, BC5CDR, BIONLP13CG)
  Stage 2: Biomedical Entity Normalization (UMLS candidates + BioBERT 10-word sliding window argmax)
  Stage 3: Entity Linking to BKGs (CODER embeddings with tau = 0.90)
  Stage 4: Semantic Relation Extraction (Hetionet & PrimeKG first-level relations)
  Stage 5: Relation Filtering (QC Filter Eq. 7 & Similarity Filter Eq. 8)
  Stage 6: Context Formatting & Atomic Non-Cutting Truncation (Section 3.6 & 4.3)
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_file import PROJECT_ROOT
from biocuep.config import BioCUEPConfig, SUPPORTED_DATASETS
from biocuep.experiments import EXPERIMENT_CONFIGS, get_experiment_config
from _1_Entities_Extraction_Normalization.Concept_Normalization import entities_normalization
from _2_Entities_Linking._1_Entities_Linking import entities_linking
from _3_Relations_Extraction._relation_extraction import relation_extraction
from _4_Filtering_Relations._1_filter_base_embedding_similarity_question import filter_relations_sm
from _4_Filtering_Relations._2_filter_base_relation_between_entities import filter_relations_qc
from _5_Relevant_Text_Explanations_Formatting._1_formating_metadata import ProcessMetaData


def run_pipeline(
    stage: str = "all",
    datasets: list = None,
    experiment: str = "exp4",
    max_tokens: int = 512
):
    target_datasets = datasets or ["BioASQ_BLURB"]
    print("=" * 70)
    print("  BioCUEP: Biomedical Context Understanding Enhancement Pipeline")
    print(f"  Target Datasets : {target_datasets}")
    print(f"  Stage to Run    : {stage}")
    print(f"  Experiment      : {experiment}")
    print(f"  Token Budget    : {max_tokens}")
    print("=" * 70)

    # Stage 1 & 2: Entity Extraction and Normalization
    if stage in ("1", "2", "all", "extract", "normalize"):
        print("\n>>> Stage 1 & 2: Entity Extraction & UMLS Normalization (BioBERT argmax)...")
        entities_normalization(target_datasets)

    # Stage 3: Entity Linking
    if stage in ("3", "all", "link"):
        print("\n>>> Stage 3: Entity Linking to BKGs via CODER (tau = 0.90)...")
        entities_linking(target_datasets)

    # Stage 4: Relation Extraction
    if stage in ("4", "all", "relations"):
        print("\n>>> Stage 4: 1st-level Relation Extraction from Hetionet & PrimeKG...")
        relation_extraction(target_datasets)

    # Stage 5: Filtering
    if stage in ("5", "all", "filter"):
        print("\n>>> Stage 5: Semantic Relation Filtering (QC Filter & TAS-B Similarity Filter)...")
        filter_relations_sm(target_datasets)
        filter_relations_qc(target_datasets)

    # Stage 6: Formatting
    if stage in ("6", "all", "format"):
        print(f"\n>>> Stage 6: Context Formatting & Atomic Truncation ({experiment})...")
        ProcessMetaData(target_datasets)

    print("\n[SUCCESS] BioCUEP pipeline run completed.")


def run_demo():
    """Runs instant demonstration on Figure 2 sample question."""
    from demo import run_figure2_demo
    run_figure2_demo()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="BioCUEP: Biomedical Context Understanding Enhancement Pipeline"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "4", "5", "6", "extract", "normalize", "link", "relations", "filter", "format"],
        help="Pipeline stage to execute (default: all)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="BioASQ_BLURB",
        choices=SUPPORTED_DATASETS + ["all"],
        help="Target biomedical QA dataset"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="exp4",
        help="Ablation configuration from Table 2 (baseline, exp1..exp13, default: exp4)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Token budget for reader input assembly (default: 512 for fine-tuned readers)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick standalone demonstration on Figure 2 test question"
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        target_ds = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]
        run_pipeline(
            stage=args.stage,
            datasets=target_ds,
            experiment=args.experiment,
            max_tokens=args.max_tokens
        )
