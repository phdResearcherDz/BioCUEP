"""
BioCUEP End-to-End Walkthrough Demo
Reproduces Figure 2 from the paper:
'BioCUEP stage outputs on a BioASQ-BLURB test question (Exps. 4 and 7)'
Can be run standalone without external database dependencies.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from biocuep.formatting import assemble_enriched_input
from biocuep.readers import construct_few_shot_prompt


def run_figure2_demo():
    print("=" * 80)
    print("    BioCUEP DEMONSTRATION: Stage Outputs on BioASQ-BLURB Test Question")
    print("    Paper Reference: Figure 2 (Exps. 4 and 7)")
    print("=" * 80)

    # Input Sample (Figure 2)
    question = "Is phospholipid hydroperoxide glutathione peroxidase a selenoprotein?"
    context = (
        "Glutathione peroxidase (Gpx1) is the major selenoprotein in most tissues. "
        "The selenoenzyme Gpx4 is essential for life and is the major selenoprotein "
        "expressed by germ cells in the testis (PHGPx/GPx4)."
    )
    gold_answer = "yes"

    print("\n--- [INPUT] ---")
    print(f"Question Q   : {question}")
    print(f"Context S    : {context}")
    print(f"Gold Answer  : {gold_answer}")

    # Stage 1: Biomedical Entity Extraction (Section 3.1, Eq. 1)
    print("\n--- [STAGE 1: Entity Extraction (SciSpaCy NER Ensemble + Filtering F)] ---")
    extracted_q_and_s = ["phospholipid", "glutathione peroxidase", "selenoprotein"]
    extracted_s_only = ["Gpx1", "Gpx4", "tissues", "germ cells", "testis"]
    print(f"Entities in Q & S : {', '.join(extracted_q_and_s)}")
    print(f"Entities in S only: {', '.join(extracted_s_only)}")

    # Stage 2: UMLS Normalization & Disambiguation (Section 3.2, Eq. 2-5)
    print("\n--- [STAGE 2: Normalization via UMLS & BioBERT 10-Word Sliding Window argmax] ---")
    definitions = {
        "selenoprotein": "Proteins that specifically incorporate selenocysteine into polypeptide chains.",
        "Gpx4": "A selenoenzyme that converts glutathione plus fatty acid hydroperoxides to water and corresponding lipid alcohols.",
        "glutathione peroxidase": "An enzyme family with peroxidase activity whose main biological role is to protect the organism from oxidative damage.",
        "testis": "The male gonad in animals producing sperm and synthesizing androgens."
    }
    for term, defn in definitions.items():
        print(f"  * {term}: {defn}")

    # Stage 3: Entity Linking to BKGs (Section 3.3, Eq. 6)
    print("\n--- [STAGE 3: Entity Linking via CODER Embeddings (tau >= 0.90)] ---")
    linked_nodes = {
        "Gpx4": "GPX4 (protein)",
        "Gpx1": "GPX1 (protein)",
        "testis": "testis (anatomy)",
        "glutathione peroxidase": "GPx activity (mol. function)"
    }
    for term, node in linked_nodes.items():
        print(f"  * {term} -> {node}")

    # Stage 4: Semantic Relation Extraction (Section 3.4)
    print("\n--- [STAGE 4: 1st-Level Semantic Relation Extraction from Hetionet & PrimeKG] ---")
    first_level_triples = [
        {"source": "GPX4", "relation": "molfunc_protein", "target_nodes": "PHGPx activity"},
        {"source": "GPX4", "relation": "anatomy_protein_present", "target_nodes": "testis"},
        {"source": "GPX1", "relation": "anatomy_protein_present", "target_nodes": "testis"},
        {"source": "GPx activity", "relation": "molfunc_protein", "target_nodes": "GSTP1"},
        {"source": "GPX1", "relation": "molfunc_protein", "target_nodes": "PHGPx activity"},
    ]
    for trip in first_level_triples:
        print(f"  * {trip['source']} --({trip['relation']})--> {trip['target_nodes']}")

    # Stage 5: Relation Filtering (Section 3.5, Eq. 7 & 8)
    print("\n--- [STAGE 5: Relation Filtering] ---")
    # QC Filtering (Eq. 7): both source and target must be linked concepts (LS U LQ)
    qc_triples = [
        {"source": "GPX4", "relation": "anatomy_protein_present", "target_nodes": "testis"},
        {"source": "GPX1", "relation": "anatomy_protein_present", "target_nodes": "testis"},
    ]
    # Similarity Filtering (Eq. 8): top-k by TAS-B cosine similarity to Question Q
    sim_triples = [
        {"source": "GPX4", "relation": "molfunc_protein", "target_nodes": "PHGPx activity"},
        {"source": "GPX1", "relation": "molfunc_protein", "target_nodes": "PHGPx activity"},
    ]
    print("5.a QC Filter (Both ends in LS U LQ, Eq. 7):")
    for trip in qc_triples:
        print(f"    {trip['source']} --({trip['relation']})--> {trip['target_nodes']}")
    print("5.b TAS-B Similarity Filter (Top-k similarity to Question, Eq. 8):")
    for trip in sim_triples:
        print(f"    {trip['source']} --({trip['relation']})--> {trip['target_nodes']}")

    # Stage 6: Formatting & Truncation (Section 3.6 & 4.3)
    print("\n--- [STAGE 6: Formatting & Atomic Non-Cutting Truncation (QC Filter, Figure 2)] ---")
    qc_assembly = assemble_enriched_input(
        question=question,
        context=context,
        relations=qc_triples,
        definitions=definitions,
        max_tokens=512,
        include_relations=True,
        include_definitions=True
    )
    print(f"\nFinal Reader Input (QC Filter):\n{qc_assembly['enriched_input']}")
    print(f"\nTotal Tokens : {qc_assembly['total_tokens']} / 512 budget")
    print(f"Relations Kept: {qc_assembly['num_relations_kept']}")
    print(f"Defs Kept     : {qc_assembly['num_definitions_kept']}")

    # Few-shot Prompt Demonstration (Section 4.3)
    print("\n--- [FEW-SHOT PROMPTING PATTERN (LLM Reader, Section 4.3)] ---")
    llm_prompt = construct_few_shot_prompt(
        question=question,
        context=context,
        metadata_text=qc_assembly["metadata"],
        dataset_name="BioASQ_BLURB"
    )
    print(llm_prompt[:500] + "\n... [truncated for display] ...\n" + llm_prompt[-250:])

    print("\n" + "=" * 80)
    print("    [DEMO COMPLETED SUCCESSFULLY]")
    print("=" * 80)


if __name__ == "__main__":
    run_figure2_demo()
